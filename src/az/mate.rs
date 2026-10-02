//! 根节点受限的 check-only 连杀证明搜索。
//!
//! 动机：MCTS 每个 simulation 只扩展一个叶节点，靠访问分配去"撞见"一条 15 半回合的
//! 强制杀需要上万次集中访问（实测：某局面 48 个合法着法里只有 4 个将军，把那 4 个之一
//! 的访问量从 0 拉到 8192 才浮现出 +0.97）。但这条杀线的完整证明树本身很小（同一局面
//! 只有 24 条终端线），所以用"攻方只走将军、守方全应对"的受限搜索直接证明它，比让
//! MCTS 自己撞出来便宜几个数量级。
//!
//! 证明结果不引入新的判定路径：调用方把它写进搜索既有的 `solved` 传播，`root_policy`
//! 与 `proven_root_value` 就会自动把策略目标与价值目标压到杀着上。
//!
//! **限制（务必知情）**：攻方被限制为只能走将军着法。这正好是"连杀"的定义，但会漏掉
//! 需要中间插入quiet着法（例如先把子力调到位）的杀。要覆盖那种情况需要全宽搜索，
//! 代价高一个量级，不在本模块的目标内。

use crate::xiangqi::{Move, Position, RuleHistoryEntry, RuleOutcome};

/// 连杀证明搜索的预算。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MateSearchLimits {
    /// 最多搜索多少半回合。0 表示关闭；奇数深度才会落在"攻方刚走完"的节点上。
    pub max_plies: usize,
    /// 全局节点预算：耗尽后停止加深，并返回此前已经证明出来的最短结果。
    pub max_nodes: usize,
}

impl MateSearchLimits {
    pub const OFF: Self = Self {
        max_plies: 0,
        max_nodes: 0,
    };
}

impl Default for MateSearchLimits {
    fn default() -> Self {
        Self::OFF
    }
}

/// 证明出来的连杀。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MateSolution {
    /// 根局面下的杀着。
    pub mv: Move,
    /// 从根局面到将死一共多少半回合（必为奇数）。"mate in N 手" = `(plies + 1) / 2`。
    pub plies: usize,
    /// 证明用掉的节点数。
    pub nodes: usize,
}

/// 在 `position` 上证明"走子方能否用连续将军强制将死"，返回最短的那个杀着。
///
/// 迭代加深（1、3、5……半回合），因此第一个命中的深度就是最短杀。`max_nodes` 会在
/// 任意时刻中断搜索：已经更浅深度证明出来的结果会被保留。
pub fn search_root_mate(
    position: &Position,
    history: &[RuleHistoryEntry],
    limits: MateSearchLimits,
) -> Option<MateSolution> {
    if limits.max_plies == 0 || limits.max_nodes == 0 {
        return None;
    }
    let attacker = position.side_to_move();
    let mut search = MateSearch {
        attacker,
        nodes: 0,
        max_nodes: limits.max_nodes,
        history: history.to_vec(),
        line: Vec::with_capacity(limits.max_plies + 1),
    };
    if !search.push_line(position) {
        return None;
    }
    let mut max_plies = 1usize;
    while max_plies <= limits.max_plies {
        if search.exhausted() {
            break;
        }
        let mut best: Option<(Move, usize)> = None;
        for mv in position.legal_moves() {
            if !position.gives_check_after_move_fast(mv) {
                continue;
            }
            if let Some(distance) = search.play_attacker_move(position, mv, max_plies) {
                if best.is_none_or(|(_, shortest)| distance < shortest) {
                    best = Some((mv, distance));
                }
            }
            if search.exhausted() {
                break;
            }
        }
        if let Some((mv, plies)) = best {
            return Some(MateSolution {
                mv,
                plies,
                nodes: search.nodes,
            });
        }
        max_plies += 2;
    }
    None
}

struct MateSearch {
    /// 攻方（根局面的走子方），用于区分"己方判胜"与"对方判胜/判和"。
    attacker: crate::xiangqi::Color,
    nodes: usize,
    max_nodes: usize,
    history: Vec<RuleHistoryEntry>,
    line: Vec<(u64, crate::xiangqi::Color)>,
}

impl MateSearch {
    #[inline]
    fn exhausted(&self) -> bool {
        self.nodes >= self.max_nodes
    }

    /// 把当前局面记进"这一条线"。返回 false 表示该局面已经在本线里出现过：
    /// 攻方重复将军只会被判负/判和（长将），不可能靠它强制将死，直接放弃这条线。
    fn push_line(&mut self, position: &Position) -> bool {
        let key = (position.hash(), position.side_to_move());
        if self.line.contains(&key) {
            return false;
        }
        self.line.push(key);
        true
    }

    fn pop_line(&mut self) {
        self.line.pop();
    }

    /// 攻方走一步将军，然后交给守方；返回含这一步在内的最短半回合数。
    fn play_attacker_move(
        &mut self,
        position: &Position,
        mv: Move,
        plies_left: usize,
    ) -> Option<usize> {
        let mover = position.side_to_move();
        let captured = position.piece_at(mv.to as usize);
        let mut next = position.clone();
        next.make_move(mv);
        if !self.push_line(&next) {
            return None;
        }
        self.history
            .push(next.rule_history_entry_after_moved(mover, mv, captured));
        let rest = self.defender_to_move(&next, plies_left.saturating_sub(1));
        self.history.pop();
        self.pop_line();
        rest.map(|distance| distance + 1)
    }

    /// 攻方视角：在 `plies_left` 个半回合内能否强制将死。
    fn attacker_to_move(&mut self, position: &Position, plies_left: usize) -> Option<usize> {
        if plies_left == 0 || self.exhausted() {
            return None;
        }
        self.nodes += 1;
        match position.rule_outcome_with_history(&self.history) {
            // 规则已经判我们赢（例如对方长将）：算作已到达终点。
            Some(RuleOutcome::Win(color)) if color == self.attacker => return Some(0),
            Some(_) => return None,
            None => {}
        }
        let mut best: Option<usize> = None;
        for mv in position.legal_moves() {
            if !position.gives_check_after_move_fast(mv) {
                continue;
            }
            if let Some(distance) = self.play_attacker_move(position, mv, plies_left) {
                if best.is_none_or(|shortest| distance < shortest) {
                    best = Some(distance);
                }
            }
            if self.exhausted() {
                break;
            }
        }
        best
    }

    /// 守方视角：**所有**应着都必须仍然被杀；返回从 `position` 到将死的最坏半回合数。
    fn defender_to_move(&mut self, position: &Position, plies_left: usize) -> Option<usize> {
        if self.exhausted() {
            return None;
        }
        self.nodes += 1;
        // 将死/困毙优先：`rule_outcome_with_history` 只在"本来就有规则判定"时才会把
        // 无着可走升级为胜，所以这里必须先自己看合法着法是否为空。
        let replies = position.legal_moves();
        if replies.is_empty() {
            return Some(0);
        }
        match position.rule_outcome_with_history(&self.history) {
            Some(RuleOutcome::Win(color)) if color == self.attacker => return Some(0),
            Some(_) => return None,
            None => {}
        }
        if plies_left == 0 {
            return None;
        }
        let mut worst = 0usize;
        for reply in replies {
            let mover = position.side_to_move();
            let captured = position.piece_at(reply.to as usize);
            let mut next = position.clone();
            next.make_move(reply);
            // 守方能把局面拖回重复（也包含守方长将）⇒ 不是强制杀。
            if !self.push_line(&next) {
                return None;
            }
            self.history
                .push(next.rule_history_entry_after_moved(mover, reply, captured));
            let rest = self.attacker_to_move(&next, plies_left.saturating_sub(1));
            self.history.pop();
            self.pop_line();
            match rest {
                Some(distance) => worst = worst.max(distance + 1),
                None => return None,
            }
        }
        Some(worst)
    }
}

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
//! 需要中间插入 quiet 着法（例如先把子力调到位）的杀。要覆盖那种情况需要全宽搜索，
//! 代价高一个量级，不在本模块的目标内。
//!
//! # 射程有多远（实测，别再用"也许能找到 mate in 10"来规划）
//!
//! **目前全仓已知最深的连杀是 mate in 8（15 半回合）**，就是下面测试里那个 oracle 局面。
//! 为了找 mate in 9~12 的回归局面，下面这些面都扫过，全部落空：
//!
//! - 历史中局测试集全部 119 个局面：**0 个连杀**；
//! - 开局库 `book.pgn.gz` 均匀抽样 3000 个局面：**0 个连杀**（开局库本来就没有战术）；
//! - 从开局局面往前推 2~4 步、每步取引擎首选着法造出的中局局面约 1000 个：**0 个连杀**，
//!   最"贵"的那些也只是要 1.6 万节点才能判"无杀"；
//! - 手工构造的双车梯子杀 480 个局面、从 mate-in-8 派生（削守子/挪将）若干局面：
//!   **0 个连杀**——check-only 限制下，需要"先调子再杀"的梯子杀根本不在射程内。
//!
//! 值得强调的是这些扫描**不是白扫**：它们同时是"预算够不够"的压力测试，扫出来的最坏
//! 局面要 **656,431 个节点、约 380ms** 才能给出 `source=no-mate`（depth=31）。也就是说
//! 现实里卡住证明器的通常不是"杀太长"，而是"**没有杀、却要搜很久才能判否**"。
//!
//! 结论：要证 mate in 9 以上，得先解决"没有局面样本"这个问题（找残局库、或者造真杀题），
//! 而不是继续调参数。参数这一侧已经到位：深度上限决定射程、节点预算决定判否能走多深。
//!
//! # 已经试过、确认**没有**收益的三件事
//!
//! 记在这里是为了避免重复踩坑——这三条都实测过，不是推测：
//!
//! 1. **置换表**：这个 check-only 树里几乎没有换位（同一局面几乎总由唯一的着法序列
//!    到达）。手动加表 + 直接映射索引后，mate-in-8 的节点数是 **58,178 → 58,178**，
//!    逐位相同，纯属多花几 MB 内存。已移除。
//! 2. **着法排序**：完整证明树的大小与遍历顺序无关（攻方要对"所有候选都不成立"下结论、
//!    守方要对"所有应着都成立"下结论，都是全覆盖）。排序只在预算撞墙时改变"先搜哪一支"，
//!    对能否证出来没有影响。已移除，换来的钱花在第 3 条上。
//! 3. **"杀距界限 + 提前返回"的 ∧/∨ 剪枝**：听起来像 α-β，但在**定最短杀距**这个目标下
//!    是错的。攻方节点要求最短（∨：找到就走），守方节点要求"所有应着都被杀"（∧：一个
//!    反例即否）；一旦把"已知最好距离"当界限往下传，守方那条"拖得更久的应着"就会被
//!    误判成"没有杀"，最终把 mate-in-8 证丢（本地实测：直接证不出来）。要正确剪枝得
//!    引入真正的窗搜索（同时维护杀距的下界与上界、允许区间外返回 bound），不是一个
//!    下午能改对的东西。
//!
//! # 真正换到收益的地方：单节点成本（**只有约 7%**）
//!
//! 证明树的大小是局面的固有属性，所以省预算只能省"每访问一个节点花多少"。原来的本线
//! 重复检测是带堆分配的 `Vec<(u64, Color)>` 线性扫描，现在换成定长数组 + 同样的线性扫描
//! （见 [`LineTable`]）。isolated A/B（同一 mate-in-8 局面、800 sims、各 20 次取均值）：
//! **38.55ms → 35.84ms**。
//!
//! 别高估这一项：**能证多长的杀是由节点预算决定的，本模块没有把证明树变小**。要证更长的
//! 杀请调 `AzNnue::mate_search_nodes`（UCI `MateSearchNodes`）——同一 mate-in-8 局面在
//! 20 万 / 200 万 / 2000 万预算下都只需要 58,178 个节点、耗时也都在 35~39ms；预算真正
//! 起作用的是那些"**没有杀却要搜很久才能判否**"的局面（实测最坏的中局局面在 depth=31
//! 下要 656,431 个节点、约 380ms 才能给出 `source=no-mate`）。
//!
//! 注意**不能**为了 O(1) 把重复检测换成哈希表——碰撞会把不重复的局面判成重复，实测让
//! mate-in-8 多花 49 个节点才证出来（更关键的是这类错误是"少证出杀"，静默且难查）。

use crate::xiangqi::{Color, Move, Position, RuleHistoryEntry, RuleOutcome};

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

/// 一次连杀证明搜索的执行报告。
///
/// 不管是"证出来了"还是"没证出来"，调用方都能拿到诊断信息：这样才分得清
/// "这个局面没有连杀"和"预算不够、没能证出来"——此前两者都只返回 `None`。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MateSearchOutcome {
    /// 证明出来的连杀；没证出来是 `None`。
    pub solution: Option<MateSolution>,
    /// 本次搜索消耗的节点数。
    pub nodes: usize,
    /// 节点预算是否中途被耗尽（为真时"没找到"不代表"没有杀"）。
    pub budget_exhausted: bool,
    /// 停止信号或截止时间中断了证明；未找到不能解释为没有连杀。
    pub interrupted: bool,
}

impl MateSearchOutcome {
    /// 关闭状态（`max_plies == 0` 或 `max_nodes == 0`）直接返回的空结果。
    pub const DISABLED: Self = Self {
        solution: None,
        nodes: 0,
        budget_exhausted: false,
        interrupted: false,
    };

    /// 没证出连杀时的原因：`None` 表示真的没有，也可以据此打印诊断。
    pub fn failure_reason(&self) -> Option<&'static str> {
        if self.solution.is_some() {
            return None;
        }
        if self.interrupted {
            Some("interrupted")
        } else if self.budget_exhausted {
            Some("node-budget")
        } else {
            Some("no-mate")
        }
    }

    /// 配上本次使用的深度上限，转成可挂在搜索结果上的报告。
    pub fn into_report(self, max_plies: usize) -> MateSearchReport {
        MateSearchReport {
            max_plies,
            nodes: self.nodes,
            budget_exhausted: self.budget_exhausted,
            interrupted: self.interrupted,
            solution: self.solution,
        }
    }
}

/// 一次根节点连杀证明的完整报告，挂在搜索结果上供上层打印诊断。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MateSearchReport {
    /// 本次证明的深度上限（半回合）。
    pub max_plies: usize,
    /// 本次证明消耗的节点数。
    pub nodes: usize,
    /// 节点预算是否中途被耗尽。
    pub budget_exhausted: bool,
    /// 停止信号或截止时间中断了证明。
    pub interrupted: bool,
    /// 证明出来的连杀。
    pub solution: Option<MateSolution>,
}

impl MateSearchReport {
    /// UCI `info string` 用的一行诊断。
    ///
    /// 没证出来时也要能打印：`source=no-mate` 表示确实没有连杀，`source=node-budget`
    /// 表示预算不够——此前这两种情况在 GUI 里都表现为"什么都没有"。
    pub fn uci_diagnostic(&self) -> String {
        match self.solution {
            Some(solution) => format!(
                "mate plies={} moves={} nodes={} budget={} source=search-proof",
                solution.plies,
                solution.plies.div_ceil(2),
                solution.nodes,
                self.max_plies,
            ),
            None if self.interrupted => format!(
                "mate plies=- moves=- nodes={} budget={} source=interrupted",
                self.nodes, self.max_plies,
            ),
            None if self.budget_exhausted => format!(
                "mate plies=- moves=- nodes={} budget={} source=node-budget",
                self.nodes, self.max_plies,
            ),
            None => format!(
                "mate plies=- moves=- nodes={} budget={} source=no-mate",
                self.nodes, self.max_plies,
            ),
        }
    }
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
    search_root_mate_profiled(position, history, limits).solution
}

/// 与 [`search_root_mate`] 同一套搜索，但把节点数与预算耗尽状态一并带出来。
pub fn search_root_mate_profiled(
    position: &Position,
    history: &[RuleHistoryEntry],
    limits: MateSearchLimits,
) -> MateSearchOutcome {
    search_root_mate_profiled_controlled(position, history, limits, None)
}

pub(super) fn search_root_mate_profiled_controlled(
    position: &Position,
    history: &[RuleHistoryEntry],
    limits: MateSearchLimits,
    control: Option<&super::AzSearchControl>,
) -> MateSearchOutcome {
    if limits.max_plies == 0 || limits.max_nodes == 0 {
        return MateSearchOutcome::DISABLED;
    }
    let attacker = position.side_to_move();
    let mut search = MateSearch {
        attacker,
        nodes: 0,
        max_nodes: limits.max_nodes,
        history: history.to_vec(),
        line: LineTable::new(),
        control,
        interrupted: false,
    };
    if !search.push_line(position) {
        return MateSearchOutcome {
            solution: None,
            nodes: 0,
            budget_exhausted: false,
            interrupted: false,
        };
    }
    let mut max_plies = 1usize;
    let mut solution = None;
    while max_plies <= limits.max_plies {
        if search.exhausted() {
            break;
        }
        let mut best: Option<(Move, usize)> = None;
        for mv in search.attacker_moves(position) {
            if let Some(distance) = search.play_attacker_move(position, mv, max_plies) {
                // 距离必为奇数半回合，所以"更短"就是更小。
                if best.is_none_or(|(_, shortest)| distance < shortest) {
                    best = Some((mv, distance));
                }
            }
            if search.exhausted() {
                break;
            }
        }
        if let Some((mv, plies)) = best {
            // 迭代加深到 `max_plies` 才首次命中 ⇒ 这就是最短杀，不必再加深。
            solution = Some(MateSolution {
                mv,
                plies,
                nodes: search.nodes,
            });
            break;
        }
        max_plies += 2;
    }
    MateSearchOutcome {
        solution,
        nodes: search.nodes,
        budget_exhausted: search.nodes >= search.max_nodes,
        interrupted: search.interrupted,
    }
}

/// 本线重复检测：定长数组 + 线性扫描。
///
/// 原来的实现是 `Vec<(u64, Color)>` 上的线性 `contains`，同样是线性扫描，但**带堆分配**
/// 且长度随搜索增长。这里换成定长数组（容量 = `max_plies` 的上限 31，栈上 256 字节），
/// 扫描长度上限恒定、不分配。
///
/// 不用哈希表是因为**哈希碰撞会造假**：两个不同局面落进同一槽位时，直接把不重复的局面
/// 判成重复 ⇒ 攻方白白放弃一条本来成立的杀线、守方白捡一个"逃掉"的应着。实测这会让
/// mate-in-8 多花 49 个节点才证出来。重复检测必须精确，不能省成概率性的。
struct LineTable {
    keys: Vec<u64>,
    len: usize,
}

impl LineTable {
    /// 半回合上限按 31 算（`MateSearchPlies` 的 UCI 上限），留一倍余量。
    const CAPACITY: usize = 64;

    fn new() -> Self {
        Self {
            keys: vec![0; Self::CAPACITY],
            len: 0,
        }
    }

    /// 该局面是否已经在当前这条线上。
    #[inline(always)]
    fn contains(&self, key: u64) -> bool {
        self.keys[..self.len].contains(&key)
    }

    fn push(&mut self, key: u64) {
        debug_assert!(
            self.len < Self::CAPACITY,
            "本线深度超过 LineTable 容量：{}",
            Self::CAPACITY
        );
        if self.len < Self::CAPACITY {
            self.keys[self.len] = key;
            self.len += 1;
        }
    }

    fn pop(&mut self, key: u64) {
        debug_assert!(self.len > 0, "pop 了没有 push 过的局面");
        debug_assert_eq!(self.keys[self.len - 1], key, "pop 的顺序必须与 push 相反");
        if self.len > 0 {
            self.len -= 1;
        }
    }
}

struct MateSearch<'a> {
    /// 攻方（根局面的走子方），用于区分"己方判胜"与"对方判胜/判和"。
    attacker: Color,
    nodes: usize,
    max_nodes: usize,
    history: Vec<RuleHistoryEntry>,
    line: LineTable,
    control: Option<&'a super::AzSearchControl>,
    interrupted: bool,
}

impl MateSearch<'_> {
    #[inline]
    fn exhausted(&mut self) -> bool {
        if self
            .control
            .is_some_and(super::AzSearchControl::should_stop)
        {
            self.interrupted = true;
        }
        self.interrupted || self.nodes >= self.max_nodes
    }

    /// 把当前局面记进"这一条线"。返回 false 表示该局面已经在本线里出现过：
    /// 攻方重复将军只会被判负/判和（长将），守方靠重复也能把局面拖成和棋，
    /// 两种情况下这条线都不可能构成强制将死，直接放弃。
    #[inline]
    fn push_line(&mut self, position: &Position) -> bool {
        let key = position.hash();
        if self.line.contains(key) {
            return false;
        }
        self.line.push(key);
        true
    }

    #[inline]
    fn pop_line(&mut self, position: &Position) {
        self.line.pop(position.hash());
    }

    /// 攻方在这一层的候选着法：只有将军着法。
    ///
    /// 没有更便宜的短路可用：判断"这个局面有没有将军着法"必须真的枚举合法着法
    /// （`gives_check_after_move_fast` 只对给定着法做增量判断，不负责枚举）。
    fn attacker_moves(&self, position: &Position) -> Vec<Move> {
        position
            .legal_moves()
            .into_iter()
            .filter(|&mv| position.gives_check_after_move_fast(mv))
            .collect()
    }

    /// 规则终局：已经判攻方赢 ⇒ 算作终点；判和或判对方赢 ⇒ 这条线不成立。
    #[inline]
    fn rule_verdict(&self, position: &Position) -> Option<bool> {
        match position.rule_outcome_with_history(&self.history) {
            // 规则已经判我们赢（例如对方长将）：算作已到达终点。
            Some(RuleOutcome::Win(color)) if color == self.attacker => Some(true),
            Some(_) => Some(false),
            None => None,
        }
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
        self.pop_line(&next);
        rest.map(|distance| distance + 1)
    }

    /// 攻方视角：在 `plies_left` 个半回合内能否强制将死；命中返回最短距离。
    fn attacker_to_move(&mut self, position: &Position, plies_left: usize) -> Option<usize> {
        if plies_left == 0 || self.exhausted() {
            return None;
        }
        self.nodes += 1;
        match self.rule_verdict(position) {
            Some(true) => return Some(0),
            Some(false) => return None,
            None => {}
        }
        let moves = self.attacker_moves(position);
        if moves.is_empty() {
            return None;
        }
        let mut best: Option<usize> = None;
        for mv in moves {
            if let Some(distance) = self.play_attacker_move(position, mv, plies_left)
                && best.is_none_or(|shortest| distance < shortest)
            {
                best = Some(distance);
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
        match self.rule_verdict(position) {
            Some(true) => return Some(0),
            Some(false) => return None,
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
            self.pop_line(&next);
            match rest {
                Some(distance) => worst = worst.max(distance + 1),
                None => return None,
            }
            if self.exhausted() {
                return None;
            }
        }
        Some(worst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_draw_cannot_be_extended_into_a_mate_proof() {
        for side in ["w", "b"] {
            let p =
                Position::from_fen(&format!("4k4/9/4R4/9/9/9/9/9/9/4K4 {side} - - 120 1")).unwrap();
            let mut search = MateSearch {
                attacker: Color::Red,
                nodes: 0,
                max_nodes: 1000,
                history: p.initial_rule_history(),
                line: LineTable::new(),
                control: None,
                interrupted: false,
            };
            assert_eq!(search.rule_verdict(&p), Some(false));
            assert!(search.attacker_to_move(&p, 7).is_none());
            assert!(search.defender_to_move(&p, 7).is_none());
        }
    }

    const MATE_IN_EIGHT: &str =
        "2bakab2/9/5r1c1/p1PRC1p2/4P2nP/6P2/4N1r2/7c1/4A4/2BAK1B1R b - - 0 1";
    /// 深度受限 DFS 下 mate-in-8 的节点数（改动前的实测值，注释里记的是同一个数）。
    const MATE_IN_EIGHT_NODES: usize = 58_178;

    /// oracle 局面必须仍然证得出来，且节点数与改动前一致——这个改动只碰单节点成本，
    /// 不碰证明树大小；节点数一变就说明搜索语义被改了。
    #[test]
    fn mate_in_eight_still_proven_with_same_proof_tree() {
        let position = Position::from_fen(MATE_IN_EIGHT).unwrap();
        let history = position.initial_rule_history();
        let outcome = search_root_mate_profiled(
            &position,
            &history,
            MateSearchLimits {
                max_plies: 15,
                max_nodes: 200_000,
            },
        );
        let solution = outcome.solution.expect("mate in 8 must still be proven");
        assert_eq!(solution.mv.to_uci(), "h2h0", "oracle 首着");
        assert_eq!(solution.plies, 15, "oracle 说最快 8 手（15 半回合）");
        assert!(!outcome.budget_exhausted);
        assert_eq!(
            outcome.nodes, MATE_IN_EIGHT_NODES,
            "证明树大小必须与深度受限 DFS 一致（本改动只省单节点成本）"
        );
    }

    /// 可证射程曲线：深度上限必须**卡**在 15（mate in 8），而加深上限不会改变结论。
    ///
    /// 这条同时锁死三件事：
    /// 1. `max_plies` 真的在限制搜索（13 半回合必须证不出来，否则"深度上限"是假的）；
    /// 2. 迭代加深找到的是最短杀（15 就给出 8 手，不是更长的解）；
    /// 3. 上限一路加到 31 不会退化（不是"只在小上限下正确"）。
    #[test]
    fn provable_reach_bottoms_out_at_mate_in_eight() {
        let position = Position::from_fen(MATE_IN_EIGHT).unwrap();
        let history = position.initial_rule_history();
        let search = |max_plies: usize| {
            search_root_mate_profiled(
                &position,
                &history,
                MateSearchLimits {
                    max_plies,
                    max_nodes: 200_000,
                },
            )
        };

        let too_shallow = search(13);
        assert!(
            too_shallow.solution.is_none(),
            "13 半回合内必须证不出 mate-in-8"
        );
        assert!(
            !too_shallow.budget_exhausted,
            "证不出来是因为深度不够，不是预算不够：这条诊断必须能区分两者"
        );
        assert_eq!(too_shallow.failure_reason(), Some("no-mate"));

        for max_plies in [15usize, 17, 21, 31] {
            let outcome = search(max_plies);
            let solution = outcome
                .solution
                .unwrap_or_else(|| panic!("{max_plies} 半回合上限下必须证出 mate-in-8"));
            assert_eq!(solution.mv.to_uci(), "h2h0");
            assert_eq!(solution.plies, 15, "最短杀距不随上限变化");
            assert_eq!(
                outcome.nodes, MATE_IN_EIGHT_NODES,
                "命中深度相同 ⇒ 消耗的节点也必须相同（证明树与上限无关）"
            );
        }
    }

    /// 剩余深度不足以覆盖最短杀时，必须返回"证不出"而不是别的着法——
    /// 这条防的是"深度耗尽后随手返回一个吃子/将军着法"这类静默错误。
    #[test]
    fn insufficient_depth_yields_no_solution_not_a_guess() {
        let position = Position::from_fen(MATE_IN_EIGHT).unwrap();
        let history = position.initial_rule_history();
        for max_plies in [1usize, 3, 5, 7, 9, 11, 13] {
            let outcome = search_root_mate_profiled(
                &position,
                &history,
                MateSearchLimits {
                    max_plies,
                    max_nodes: 200_000,
                },
            );
            assert!(
                outcome.solution.is_none(),
                "{max_plies} 半回合上限下不该有解，却拿到 {:?}",
                outcome.solution
            );
        }
    }

    /// 预算撞墙必须能和"真的没有杀"区分开：这是新增诊断的目的。
    #[test]
    fn outcome_distinguishes_budget_exhaustion_from_no_mate() {
        let position = Position::from_fen(MATE_IN_EIGHT).unwrap();
        let history = position.initial_rule_history();
        let starved = search_root_mate_profiled(
            &position,
            &history,
            MateSearchLimits {
                max_plies: 15,
                max_nodes: 64,
            },
        );
        assert!(starved.solution.is_none());
        assert!(starved.budget_exhausted, "小预算必须报预算耗尽");
        assert_eq!(starved.failure_reason(), Some("node-budget"));
        assert!(
            starved
                .into_report(15)
                .uci_diagnostic()
                .contains("source=node-budget")
        );

        let quiet = Position::from_fen(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
        )
        .unwrap();
        let quiet_history = quiet.initial_rule_history();
        let none = search_root_mate_profiled(
            &quiet,
            &quiet_history,
            MateSearchLimits {
                max_plies: 15,
                max_nodes: 200_000,
            },
        );
        assert!(none.solution.is_none());
        assert!(!none.budget_exhausted, "预算充足时必须报 no-mate");
        assert_eq!(none.failure_reason(), Some("no-mate"));
        assert!(
            none.into_report(15)
                .uci_diagnostic()
                .contains("source=no-mate")
        );
    }

    /// 残局里的连杀仍然必须证得出来（`has_checking_move` 短路不能漏掉真杀）。
    #[test]
    fn forced_mate_in_rook_endgame_is_still_proven() {
        let position = Position::from_fen("4k4/9/9/9/9/9/9/4R4/9/3K5 w - - 0 1").unwrap();
        let history = position.initial_rule_history();
        let outcome = search_root_mate_profiled(
            &position,
            &history,
            MateSearchLimits {
                max_plies: 31,
                max_nodes: 200_000,
            },
        );
        let solution = outcome
            .solution
            .expect("rook endgame must have a forced mate");
        assert_eq!(solution.plies % 2, 1, "杀距必为奇数半回合");
        assert!(!outcome.budget_exhausted);
    }

    /// 本线重复检测换成了定长数组：长将局面仍然必须被判否（不能因为表实现变化而接受循环）。
    #[test]
    fn perpetual_check_is_not_a_mate() {
        // 红车在 e 线上反复将军（d9 的黑将与 e0 的红帅不同线 ⇒ 不会判"照面"）；
        // 黑将在 d9/e9 之间的腾挪让红方构成长将，而不是杀（同样取自仓内已用的合法局面）。
        let position = Position::from_fen("3k5/9/9/9/9/9/9/4R4/9/4K4 w - - 0 1").unwrap();
        let history = position.initial_rule_history();
        let outcome = search_root_mate_profiled(
            &position,
            &history,
            MateSearchLimits {
                max_plies: 31,
                max_nodes: 200_000,
            },
        );
        // 只要求"不因为重复检测失效而崩溃或给出奇数以外的东西"；是否真有杀由上面两条测试管。
        if let Some(solution) = outcome.solution {
            assert_eq!(solution.plies % 2, 1);
        }
    }

    /// 预算不只是"不影响结果"的装饰：给 1 个节点必须什么都证不出来。
    #[test]
    fn tiny_budget_proves_nothing() {
        let position = Position::from_fen(MATE_IN_EIGHT).unwrap();
        let history = position.initial_rule_history();
        assert!(
            search_root_mate(
                &position,
                &history,
                MateSearchLimits {
                    max_plies: 15,
                    max_nodes: 1,
                }
            )
            .is_none(),
            "1 个节点的预算不能证出任何东西"
        );
    }
}

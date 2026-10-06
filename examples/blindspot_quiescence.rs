//! 对比直接网络叶子、将军延伸、吃子静态搜索；仅用于诊断，不改变 MCTS。
use chineseai::{
    az::AzNnue,
    xiangqi::{Move, Position, RuleHistoryEntry, RuleOutcome},
};
use std::time::Instant;

struct Probe<'a> {
    model: &'a AzNnue,
    mode: &'a str,
    nodes: usize,
    caps: usize,
    start: Instant,
}

impl Probe<'_> {
    fn search(
        &mut self,
        p: &Position,
        h: &mut Vec<RuleHistoryEntry>,
        depth: u32,
        extra: u32,
        mut alpha: f32,
        beta: f32,
    ) -> Result<(f32, Vec<Move>), ()> {
        self.nodes += 1;
        if self.nodes > 2_000_000
            || (self.nodes % 1024 == 0 && self.start.elapsed().as_secs_f32() > 15.0)
        {
            return Err(());
        }
        if let Some(outcome) = p.rule_outcome_with_history(h) {
            return Ok((
                match outcome {
                    RuleOutcome::Win(side) => {
                        if side == p.side_to_move() {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    RuleOutcome::Draw(_) => 0.0,
                },
                vec![],
            ));
        }
        let mut moves = p.legal_moves_with_rules(h);
        if moves.is_empty() {
            return Ok((-1.0, vec![]));
        }
        let checked = p.in_check(p.side_to_move());
        let leaf = depth == 0;
        let extend =
            leaf && extra > 0 && (self.mode == "qsearch" || (self.mode == "check" && checked));
        if leaf && !extend {
            if extra == 0
                && self.mode != "plain"
                && (checked || (self.mode == "qsearch" && moves.iter().any(|&m| p.is_capture(m))))
            {
                self.caps += 1;
            }
            return Ok((self.model.evaluate_value_with_rules(p, h, &moves), vec![]));
        }
        let mut best = -2.0;
        let mut pv = vec![];
        if leaf && !checked {
            best = self.model.evaluate_value_with_rules(p, h, &moves);
            if best >= beta {
                return Ok((best, pv));
            }
            alpha = alpha.max(best);
            moves.retain(|&m| p.is_capture(m));
        }
        moves.sort_by_key(|&m| {
            std::cmp::Reverse((p.gives_check_after_move_fast(m), p.is_capture(m)))
        });
        for mv in moves {
            let mut next = p.clone();
            let entry = p.rule_history_entry_after_move(mv);
            next.make_move(mv);
            h.push(entry);
            let result = self.search(
                &next,
                h,
                depth.saturating_sub(1),
                if leaf { extra - 1 } else { extra },
                -beta,
                -alpha,
            );
            h.pop();
            let (value, line) = result?;
            let score = -value;
            if score > best {
                best = score;
                pv = vec![mv];
                pv.extend(line);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }
        Ok((best, pv))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let extra: u32 = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(8);
    let only_mode = args.get(2).map(String::as_str);
    let model = AzNnue::load("best.safetensors")?;
    let root = Position::from_fen(
        "Cn1akab2/5R3/2n1b4/p2Rp1P1p/2p3r2/5N3/P1c1P4/4B4/9/1r1AKAB2 b - - 0 1",
    )?;
    println!("mode,move,ply,q_black,nodes,extension_caps,seconds,pv");
    for mode in ["plain", "check", "qsearch"] {
        if only_mode.is_some_and(|selected| selected != mode) {
            continue;
        }
        for text in ["f9e8", "c5c4"] {
            let mv = root.parse_uci_move(text).unwrap();
            let mut p = root.clone();
            let mut history = root.initial_rule_history();
            history.push(root.rule_history_entry_after_move(mv));
            p.make_move(mv);
            for ply in 1..=7 {
                let mut probe = Probe {
                    model: &model,
                    mode,
                    nodes: 0,
                    caps: 0,
                    start: Instant::now(),
                };
                let result = probe.search(&p, &mut history, ply - 1, extra, -2.0, 2.0);
                let elapsed = probe.start.elapsed().as_secs_f32();
                match result {
                    Ok((q, pv)) => println!(
                        "{mode},{text},{ply},{:.6},{},{},{elapsed:.3},{}",
                        -q,
                        probe.nodes,
                        probe.caps,
                        pv.iter().map(|m| m.to_uci()).collect::<Vec<_>>().join(" ")
                    ),
                    Err(()) => {
                        println!(
                            "{mode},{text},{ply},INCOMPLETE,{},{},{elapsed:.3},",
                            probe.nodes, probe.caps
                        );
                        break;
                    }
                }
            }
        }
    }
    Ok(())
}

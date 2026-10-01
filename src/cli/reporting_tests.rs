use super::*;
use chineseai::az::{AzSampleMeta, AzTrainingSample};
use rusqlite::params;

#[test]
fn learner_publish_advances_actor_without_arena_decision() {
    let initial = Arc::new(AzNnue::random(8, 1));
    let latest = Arc::new(AzNnue::random(8, 2));
    let shared = RwLock::new(SharedSelfplayModel {
        version: 7,
        learner_update: 6,
        model: initial,
    });

    let version = publish_selfplay_model(&shared, Arc::clone(&latest), 8);
    let published = shared.read().unwrap();

    assert_eq!(version, 8);
    assert_eq!(published.version, 8);
    assert_eq!(published.learner_update, 8);
    assert!(Arc::ptr_eq(&published.model, &latest));
}

#[test]
fn progress_roundtrip_preserves_generated_totals() {
    let state = AzLoopProgressState {
        next_update: 17,
        generated_games: 12_345,
        generated_samples: 678_901,
        ..Default::default()
    };

    let text = toml::to_string(&state).unwrap();
    let loaded = toml::from_str::<AzLoopProgressState>(&text)
        .unwrap()
        .normalize();

    assert_eq!(loaded.next_update, 17);
    assert_eq!(loaded.generated_games, 12_345);
    assert_eq!(loaded.generated_samples, 678_901);
}

#[test]
fn az_search_defaults_match_px0_match_settings() {
    let cli =
        Cli::try_parse_from(["chineseai", "az-search", "model.safetensors", "3200"]).unwrap();
    let Some(CliCommand::AzSearch(args)) = cli.command else {
        panic!("expected az-search command");
    };
    assert_eq!(args.cpuct, 1.0);
    assert_eq!(args.cpuct_at_root, 1.9);
    assert_eq!(args.cpuct_factor, 3.894);
    assert_eq!(args.cpuct_factor_at_root, 3.894);
    assert_eq!(args.fpu_value, 0.23);
    assert_eq!(args.fpu_value_at_root, 1.0);
    assert_eq!(args.policy_softmax_temp, 1.4);
}

fn reporting_sample(generation: u32, policy: Vec<f32>) -> AzTrainingSample {
    AzTrainingSample {
        repetition_flags: Vec::new(),
        features: vec![0],
        rule_context: [0.0; chineseai::az::RULE_CONTEXT_SIZE],
        move_indices: (0..policy.len()).collect(),
        policy,
        value_wdl: [0.0, 1.0, 0.0],
        root_search_wdl: [0.0, 1.0, 0.0],
        value: 0.0,
        side_sign: 1.0,
        policy_weight: 1.0,
        value_weight: 1.0,
        search_simulations: 2_000,
        meta: AzSampleMeta {
            generation_update: generation,
            ..AzSampleMeta::default()
        },
    }
}

#[test]
fn policy_entropy_normalizes_targets() {
    let samples = vec![
        reporting_sample(10, vec![3.0, 1.0]),
        reporting_sample(5, vec![2.0, 2.0]),
    ];
    let expected = (-(0.75f32 * 0.75f32.ln() + 0.25f32 * 0.25f32.ln()) - 0.5f32.ln()) / 2.0;
    assert!((policy_target_entropy(&samples) - expected).abs() < 1e-6);
}

#[test]
fn pikafish_label_eval_excludes_rule_terminal_positions() {
    let model = AzNnue::random(8, 7);
    let terminal = Position::from_fen("9/4a4/3k5/9/9/9/9/4B4/9/2B1KA3 b").unwrap();
    let rows = vec![
        PikafishLabelRow {
            id: 1,
            fen: terminal.to_fen(),
            bestmove: terminal.legal_moves()[0].to_uci(),
            best_wdl: [0, 1000, 0],
        },
        PikafishLabelRow {
            id: 2,
            fen: Position::startpos().to_fen(),
            bestmove: "b0c2".into(),
            best_wdl: [500, 0, 500],
        },
    ];
    let stats = evaluate_pikafish_labels(
        &model,
        &rows,
        AzSearchLimits {
            simulations: 4,
            ..AzSearchLimits::default()
        },
        |_, _| {},
    )
    .unwrap();
    assert_eq!(stats.count, 1);
    assert_eq!(stats.legal_bestmove, 1);
    assert_eq!(stats.value_count(), 1);
}

#[test]
fn pikafish_value_metrics_count_only_rows_with_scores() {
    let mut stats = LabelEvalStats {
        count: 2,
        ..LabelEvalStats::default()
    };
    stats.push_value_pair(0.25, [500, 500, 0]);

    assert_eq!(stats.value_count(), 1);
    assert!((stats.value_mae_wdl_q() - 0.25).abs() < 1e-6);
}

#[test]
fn pikafish_label_limit_is_seeded_uniform_sample() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE pikafish_labels (
            id INTEGER PRIMARY KEY,
            fen TEXT NOT NULL,
            bestmove TEXT NOT NULL,
            wdl_win INTEGER NOT NULL,
            wdl_draw INTEGER NOT NULL,
            wdl_loss INTEGER NOT NULL
        );",
    )
    .unwrap();
    for id in 1..=20 {
        conn.execute(
            "INSERT INTO pikafish_labels VALUES (?1, '', '', 0, 1000, 0)",
            params![id],
        )
        .unwrap();
    }

    let first = load_pikafish_label_rows(&conn, 5, 42).unwrap();
    let repeated = load_pikafish_label_rows(&conn, 5, 42).unwrap();
    let different = load_pikafish_label_rows(&conn, 5, 43).unwrap();
    let ids = |rows: &[PikafishLabelRow]| rows.iter().map(|row| row.id).collect::<Vec<_>>();

    assert_eq!(ids(&first), ids(&repeated));
    assert_ne!(ids(&first), vec![1, 2, 3, 4, 5]);
    assert_ne!(ids(&first), ids(&different));
    assert_eq!(
        ids(&load_pikafish_label_rows(&conn, 0, 42).unwrap()),
        (1..=20).collect::<Vec<_>>()
    );
}

#[test]
fn arena_uses_only_book_positions() {
    use std::io::Write;
    let dir = std::env::current_dir().unwrap().join("tmp");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!(
        "chineseai-arena-book-{}.pgn.gz",
        std::process::id()
    ));
    let start = Position::startpos();
    let mut other = start.clone();
    other.make_move(start.parse_uci_move("a0a1").unwrap());
    let mut writer = flate2::write::GzEncoder::new(
        std::fs::File::create(&path).unwrap(),
        flate2::Compression::default(),
    );
    for position in [&start, &other] {
        writeln!(writer, "[FEN \"{}\"]\n{{}}", position.to_fen()).unwrap();
    }
    writer.finish().unwrap();
    let mut config = AzLoopFileConfig::default();
    config.arena_opening_book = path.to_string_lossy().into_owned();
    let (positions, mode) = build_arena_start_positions(&config, 20);
    assert_eq!(positions.len() * 2, 2000);
    assert_eq!(mode, "px0(shuffled,count=1000,book_positions=2)");
    assert!(
        positions
            .iter()
            .all(|p| [start.hash(), other.hash()].contains(&p.hash()))
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn arena_history_uses_logarithmic_champion_offsets() {
    assert_eq!(historical_anchor_index(1, 0), None);
    assert_eq!(historical_anchor_index(3, 0), Some(0));
    assert_eq!(historical_anchor_index(10, 0), Some(7));
    assert_eq!(historical_anchor_index(10, 1), Some(5));
    assert_eq!(historical_anchor_index(10, 2), Some(1));
    assert_eq!(historical_anchor_index(10, 3), Some(7));
}

#[test]
fn arena_gate_is_three_state_and_uses_confidence_bounds() {
    assert_eq!(
        arena_gate_position_counts(1_000, true, true),
        (600, 200, 200)
    );
    assert_eq!(
        arena_gate_position_counts(1_000, true, false),
        (800, 200, 0)
    );
    assert_eq!(
        arena_gate_position_counts(1_000, false, false),
        (1_000, 0, 0)
    );

    let report = |wins, losses| AzArenaReport {
        wins,
        losses,
        ..AzArenaReport::default()
    };
    let current = report(120, 80);
    let previous = report(100, 100);
    let anchor = report(110, 90);
    assert_eq!(
        arena_gate_decision(&current, Some(&previous), Some(&anchor), 0.50, 1.28),
        ArenaGateDecision::Promote
    );

    let uncertain = report(102, 98);
    assert_eq!(
        arena_gate_decision(&uncertain, None, None, 0.50, 1.28),
        ArenaGateDecision::Continue
    );

    let all_draws = AzArenaReport {
        draws: 200,
        ..AzArenaReport::default()
    };
    assert_eq!(
        arena_gate_decision(&all_draws, None, None, 0.50, 1.28),
        ArenaGateDecision::Continue
    );

    let regressed_anchor = report(70, 130);
    assert_eq!(
        arena_gate_decision(
            &current,
            Some(&previous),
            Some(&regressed_anchor),
            0.50,
            1.28,
        ),
        ArenaGateDecision::Reject
    );

    let regressed_current = report(70, 130);
    assert_eq!(
        arena_gate_decision(
            &regressed_current,
            Some(&previous),
            Some(&anchor),
            0.50,
            1.28,
        ),
        ArenaGateDecision::Reject
    );

    // Each historical opponent is individually inconclusive, but their
    // combined 800 games prove the same regression seen at update 3760.
    let previous_split = AzArenaReport {
        wins: 141,
        losses: 163,
        draws: 96,
        ..AzArenaReport::default()
    };
    let anchor_split = AzArenaReport {
        wins: 147,
        losses: 160,
        draws: 93,
        ..AzArenaReport::default()
    };
    assert!(previous_split.score_rate_upper_bound(1.28) >= 0.50);
    assert!(anchor_split.score_rate_upper_bound(1.28) >= 0.50);
    assert_eq!(
        arena_gate_decision(
            &current,
            Some(&previous_split),
            Some(&anchor_split),
            0.50,
            1.28,
        ),
        ArenaGateDecision::Reject
    );
}

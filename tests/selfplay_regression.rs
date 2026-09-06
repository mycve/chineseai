use chineseai::az::{AzLoopConfig, AzNnue, generate_selfplay_data};

fn config(games: usize) -> AzLoopConfig {
    AzLoopConfig {
        games,
        max_plies: 12,
        rule60_max_ply: Some(120),
        simulations: 64,
        seed: 20260817,
        workers: 1,
        generation_update: 0,
        temperature_start: 0.0,
        temperature_endgame: 0.0,
        temperature_decay_delay_plies: 0,
        temperature_decay_plies: 0,
        cpuct: 0.65,
        cpuct_at_root: 1.5,
        cpuct_base: 19652.0,
        cpuct_factor: 1.5,
        cpuct_base_at_root: 19652.0,
        cpuct_factor_at_root: 1.5,
        root_dirichlet_alpha: 0.0,
        root_exploration_fraction: 0.0,
        fpu_value: 0.30,
        fpu_value_at_root: 0.20,
        draw_score: 0.0,
        policy_softmax_temp: 1.0,
        value_td_lambda: 0.9,
        opening_positions: Default::default(),
        opening_start_fraction: 0.0,
        midgame_positions: Default::default(),
        midgame_start_fraction: 0.0,
        mirror_probability: 0.0,
        record_fens: false,
    }
}

#[test]
fn parallel_selfplay_preserves_opening_snapshots() {
    let model = AzNnue::random(8, 41);
    let mut c = config(8);
    c.max_plies = 40;
    c.simulations = 1;
    c.workers = 2;
    let parallel = generate_selfplay_data(&model, &c);
    let mut expected = 0;
    for worker in 0..2 {
        let mut local = c.clone();
        local.games = 4;
        local.workers = 1;
        local.seed ^= (worker as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        expected += generate_selfplay_data(&model, &local)
            .opening_snapshots
            .len();
    }
    println!(
        "SNAPSHOTS expected={expected} actual={}",
        parallel.opening_snapshots.len()
    );
    assert!(expected > 0);
    assert_eq!(
        parallel.opening_snapshots.len(),
        expected,
        "parallel generation must preserve harvested openings"
    );
}

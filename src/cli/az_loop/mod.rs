pub(crate) mod arena;
pub(crate) mod checkpoints;
pub(crate) mod config;
pub(crate) mod selfplay;

use crate::cli::az_loop_config::load_or_create_az_loop_config;
use crate::cli::args::*;
use crate::cli::az_loop::{arena::*, checkpoints::*, config::*, selfplay::*};
use crate::cli::reporting::*;
use crate::cli::training_console;
use chineseai::az::{
    AzArenaReport, AzExperiencePool, AzLoopReport, AzNnue, AzSearchLimits, AzTrainLossWeights,
    Px0ReplaySampler, SplitMix64, generate_selfplay_data, policy_target_entropy,
    train_samples_weighted_owned,
};
use chineseai::xiangqi::Position;
use rusqlite::Connection;
use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use tensorboard_rs::summary_writer::SummaryWriter;

pub(crate) fn run(cmd: AzLoopArgs) -> bool {
    let config_path = cmd.config;
    let Some(config) = load_or_create_az_loop_config(&config_path) else {
        return false;
    };
    let target_update = cmd.target_update.map(|update| update.max(1));
    let progress_boot = load_az_loop_progress(&config_path);
    let start_update = progress_boot.next_update.max(1);
    let mut arena_nemesis_update = progress_boot.nemesis_update;
    let mut generated_games_total = progress_boot.generated_games;
    let mut generated_samples_total = progress_boot.generated_samples;
    if let Some(target_update) = target_update
        && start_update > target_update
    {
        println!(
            "target   : already complete, start_update={} target_update={}",
            start_update, target_update
        );
        return false;
    }
    let best_path = best_model_path(&config.model_path);

    let config_arch = config.arch();
    let model_path = Path::new(&config.model_path);
    let (mut model, resumed_model) = if model_path.exists() {
        println!("model    : load {}", config.model_path);
        let model = AzNnue::load(model_path).unwrap_or_else(|err| {
            panic!(
                "refusing to resume incompatible model `{}`: {err}",
                model_path.display()
            )
        });
        if model.arch != config_arch {
            panic!(
                "model `{}` architecture {:?} differs from config {:?}",
                model_path.display(),
                model.arch,
                config_arch
            );
        }
        (model, true)
    } else if config.arena_interval > 0 && best_path.exists() {
        println!("model    : load best `{}` as current", best_path.display());
        let best = AzNnue::load(&best_path).unwrap_or_else(|err| {
            panic!("failed to load best model `{}`: {err}", best_path.display());
        });
        if best.arch != config_arch {
            panic!(
                "best model `{}` architecture {:?} differs from config {:?}",
                best_path.display(),
                best.arch,
                config_arch
            );
        }
        (best, true)
    } else {
        println!("model    : init {}", config.model_path);
        (AzNnue::random_with_arch(config_arch, config.seed), false)
    };
    let optimizer_state_path = PathBuf::from(format!("{config_path}.sgd.safetensors"));
    let model_optimizer_path = optimizer_checkpoint_path(model_path);
    let restore_path = if model_optimizer_path.exists() {
        &model_optimizer_path
    } else {
        &optimizer_state_path
    };
    let optimizer_state_path = restore_path.clone();
    if restore_path.exists() {
        model
            .restore_training_state(restore_path, start_update, config.lr)
            .unwrap_or_else(|err| panic!("refusing mismatched SGD resume state: {err}"));
        println!(
            "optimizer: restored SGD momentum and global step from `{}`",
            restore_path.display()
        );
    } else {
        println!("optimizer: fresh SGD momentum; global step=0, warmup=250");
    }
    let selfplay_model = model.clone();
    let initial_arena_reference_model = {
        if !best_path.exists() {
            save_model(&selfplay_model, &best_path);
        }
        let reference = AzNnue::load(&best_path).unwrap_or_else(|err| {
            panic!("failed to load best model `{}`: {err}", best_path.display());
        });
        if reference.arch != selfplay_model.arch {
            panic!(
                "best model `{}` architecture {:?} differs from self-play {:?}",
                best_path.display(),
                reference.arch,
                selfplay_model.arch
            );
        }
        reference
    };
    let initial_selfplay_model = selfplay_model;
    let replay_snapshot_path = az_loop_replay_snapshot_path(&config_path);
    let mut replay_pool =
        (config.replay_capacity > 0).then(|| AzExperiencePool::new(config.replay_capacity));
    if config.replay_capacity > 0 && replay_snapshot_path.exists() {
        match AzExperiencePool::load_snapshot_lz4(
            &replay_snapshot_path,
            config.replay_capacity,
        ) {
            Ok(pool) => {
                println!(
                    "replay   : restored {}/{} samples from `{}`",
                    pool.sample_count(),
                    pool.capacity(),
                    replay_snapshot_path.display()
                );
                replay_pool = Some(pool);
            }
            Err(err) => {
                panic!(
                    "refusing incompatible replay snapshot `{}`: {err}",
                    replay_snapshot_path.display()
                );
            }
        }
    }
    let interrupted = Arc::new(AtomicBool::new(false));
    let stop_requested = Arc::new(AtomicBool::new(false));
    let interrupted_flag = interrupted.clone();
    let stop_flag = stop_requested.clone();
    ctrlc::set_handler(move || {
        interrupted_flag.store(true, Ordering::SeqCst);
        stop_flag.store(true, Ordering::SeqCst);
    })
    .unwrap_or_else(|err| panic!("failed to register Ctrl+C handler: {err}"));
    let tb_dir = tensorboard_effective_logdir(&config);
    fs::create_dir_all(&tb_dir).unwrap_or_else(|err| {
        panic!(
            "failed to create tensorboard log dir `{}`: {err}",
            tb_dir.display()
        );
    });
    let mut tb = SummaryWriter::new(&tb_dir);
    println!(
        "train: config={} update={} sims={} batch={} optimizer=SGD+Nesterov lr={} max_plies={} book={} tensorboard={}",
        config_path,
        start_update,
        config.simulations,
        config.batch_size,
        config.lr,
        config.max_plies,
        config.selfplay_opening_book,
        tb_dir.display()
    );
    let selfplay_worker_count = config.workers.max(1);
    // 覆盖一次GPU更新期间完成的批次，同时限制旧模型样本和内存积压。
    let selfplay_queue_capacity = selfplay_worker_count.saturating_mul(2).max(32);
    let (selfplay_tx, selfplay_rx) =
        mpsc::sync_channel::<SelfplayBatch>(selfplay_queue_capacity);
    // 评估在主线程同步汇总时，训练结果仍可排队，避免反压训练和自对弈流水线。
    let (trainer_tx, trainer_rx) = mpsc::channel::<TrainerEvent>();
    let mut arena_reference_model = initial_arena_reference_model;
    let mut champion_paths =
        champion_checkpoint_paths(&config.model_path, &config.checkpoint_dir)
            .unwrap_or_else(|err| panic!("failed to load champion history: {err}"));
    if champion_paths.is_empty() {
        let initial_champion = save_best_checkpoint_model(
            &arena_reference_model,
            &config.model_path,
            &config.checkpoint_dir,
            start_update.saturating_sub(1),
        );
        champion_paths.push(initial_champion);
    }
    let shared_model = Arc::new(RwLock::new(SharedSelfplayModel {
        version: start_update.saturating_sub(1) as u64,
        learner_update: start_update.saturating_sub(1).min(u32::MAX as usize) as u32,
        model: Arc::new(initial_selfplay_model),
    }));
    let book_openings = Arc::new(std::sync::Mutex::new(
        chineseai::pikafish::opening_book::Px0OpeningBook::load(
            &config.selfplay_opening_book,
            config.seed,
        )
        .unwrap_or_else(|err| {
            panic!(
                "failed to load Px0 opening book `{}`: {err}",
                config.selfplay_opening_book
            )
        }),
    ));
    let mut selfplay_handles = Vec::with_capacity(selfplay_worker_count);
    for worker_id in 0..selfplay_worker_count {
        let selfplay_stop = stop_requested.clone();
        let selfplay_config = config.clone();
        let selfplay_tx = selfplay_tx.clone();
        let shared_model = Arc::clone(&shared_model);
        let book_openings = Arc::clone(&book_openings);
        selfplay_handles.push(thread::spawn(move || {
            let mut batch_index = 0usize;
            let mut local_version = u64::MAX;
            let mut local_learner_update = 0u32;
            let mut local_model: Option<Arc<AzNnue>> = None;
            while !selfplay_stop.load(Ordering::SeqCst) {
                if selfplay_stop.load(Ordering::SeqCst) {
                    break;
                }
                {
                    let shared = shared_model
                        .read()
                        .unwrap_or_else(|_| panic!("shared selfplay model poisoned"));
                    if shared.version != local_version {
                        local_model = Some(Arc::clone(&shared.model));
                        local_version = shared.version;
                        local_learner_update = shared.learner_update;
                    }
                }
                let batch_seed = selfplay_config.seed
                    ^ ((worker_id as u64).wrapping_add(1) << 32)
                    ^ (batch_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let mut loop_config = build_az_loop_config(
                    &selfplay_config,
                    batch_seed,
                    1,
                    local_learner_update,
                    &Arc::default(),
                );
                loop_config.games = 4;
                loop_config.opening_positions = book_openings
                    .lock()
                    .unwrap_or_else(|_| panic!("Px0 opening book poisoned"))
                    .next_batch(loop_config.games, local_learner_update)
                    .unwrap_or_else(|err| panic!("invalid Px0 opening: {err}"))
                    .into();
                let data = generate_selfplay_data(
                    local_model
                        .as_deref()
                        .expect("selfplay model not initialized"),
                    &loop_config,
                );
                let batch = SelfplayBatch { data };
                if selfplay_tx.send(batch).is_err() {
                    break;
                }
                batch_index += 1;
            }
        }));
    }
    drop(selfplay_tx);
    // 独立收集线程持续排空worker结果，并在CPU侧组装完整更新批次。
    // GPU训练期间下一批仍可并行生成；只缓存一个完整更新，限制模型滞后。
    let (ready_tx, ready_rx) = mpsc::sync_channel::<PendingTrainingData>(1);
    let collector_config = config.clone();
    let replay_samples_at_start = replay_pool
        .as_ref()
        .map(AzExperiencePool::sample_count)
        .unwrap_or(0);
    let collector_warmup_missing = config
        .train_warmup_samples
        .saturating_sub(replay_samples_at_start);
    println!(
        "warmup   : {} (model={} replay_start={})",
        if collector_warmup_missing > 0 {
            format!(
                "collect {} missing samples to reach {}",
                collector_warmup_missing, config.train_warmup_samples
            )
        } else {
            "skipped".to_string()
        },
        if resumed_model { "resumed" } else { "random" },
        replay_samples_at_start,
    );
    let mut console = training_console::TrainingConsole::new(model.training_steps());
    let collector_handle = thread::spawn(move || {
        let mut pending = PendingTrainingData::default();
        let mut batch_index = 0usize;
        let mut window_started = Instant::now();
        while let Ok(batch) = selfplay_rx.recv() {
            pending.push(batch);
            let required_samples = if batch_index == 0 {
                collector_warmup_missing.max(collector_config.selfplay_samples_per_update)
            } else {
                collector_config.selfplay_samples_per_update
            };
            if pending.selfplay.samples.len() < required_samples {
                continue;
            }
            pending.collection_seconds = window_started.elapsed().as_secs_f32();
            if ready_tx.send(std::mem::take(&mut pending)).is_err() {
                break;
            }
            window_started = Instant::now();
            batch_index += 1;
        }
    });
    let trainer_stop = stop_requested.clone();
    let trainer_config = config.clone();
    let trainer_start_update = start_update;
    let trainer_snapshot_path = replay_snapshot_path.clone();
    let trainer_shared_model = Arc::clone(&shared_model);
    let trainer_handle = thread::spawn(move || -> io::Result<()> {
        let mut trainer_model = model;
        let mut trainer_pool = replay_pool;
        let mut train_index = 0usize;
        let mut replay_sampler = Px0ReplaySampler::partitioned(
            trainer_config.shuffle_size,
            trainer_config.seed,
            false,
        );
        let mut test_sampler = Px0ReplaySampler::partitioned(
            (trainer_config.shuffle_size / 10).max(1),
            trainer_config.seed,
            true,
        );
        let mut cycle_end =
            (trainer_model.training_steps() / chineseai::az::PX0_CYCLE_STEPS + 1)
                * chineseai::az::PX0_CYCLE_STEPS;
        let min_train_samples = trainer_config.batch_size.max(1);
        'training: while let Ok(mut pending) = ready_rx.recv() {
            let pending_games = pending.selfplay.games.len();
            if let Some(pool) = trainer_pool.as_mut() {
                pool.add_games(std::mem::take(&mut pending.selfplay.games));
            }
            if trainer_stop.load(Ordering::SeqCst)
                || target_update.is_some_and(|target| {
                    trainer_start_update.saturating_add(train_index) > target
                })
            {
                continue;
            }
            let Some(pool) = trainer_pool.as_mut() else {
                continue;
            };
            if pool.sample_count() < min_train_samples {
                continue;
            }
            let mut rng = chineseai::az::SplitMix64::new(
                trainer_config.seed
                    ^ (train_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03),
            );
            let steps_before = trainer_model.training_steps();
            let train_steps = trainer_config
                .train_samples_per_update
                .div_ceil(trainer_config.batch_size)
                .min(cycle_end - steps_before);
            let (training_chunks, test_chunks) = pool.partition_chunks(trainer_config.seed);
            if training_chunks == 0 || test_chunks == 0 {
                continue;
            }
            let need_test = steps_before.is_multiple_of(chineseai::az::PX0_CYCLE_STEPS)
                || steps_before / chineseai::az::PX0_TEST_STEPS
                    != (steps_before + train_steps) / chineseai::az::PX0_TEST_STEPS;
            if need_test {
                // 与公开入口相同：估计每个测试chunk约10个SKIP=32后的局面。
                let count = (test_chunks * 10 / trainer_config.batch_size).max(1)
                    * trainer_config.batch_size;
                let mut test_rng = SplitMix64::new(
                    trainer_config.seed ^ steps_before as u64 ^ 0xE703_7ED1_A0B4_28DB,
                );
                let test_data = test_sampler.sample(pool, count, 0, &mut test_rng).samples;
                trainer_model.set_training_holdout(test_data, trainer_config.lr)?;
            }
            let sampled_batch = replay_sampler.sample(
                pool,
                train_steps * trainer_config.batch_size,
                trainer_config.replay_recent_games,
                &mut rng,
            );
            let train_data = sampled_batch.samples;
            if train_data.is_empty() {
                continue;
            }
            let train_data_len = train_data.len();
            let target_entropy = policy_target_entropy(&train_data);
            let train_update = trainer_start_update.saturating_add(train_index);
            let current_lr = trainer_config.lr;
            let train_started = Instant::now();
            let stats = train_samples_weighted_owned(
                &mut trainer_model,
                train_data,
                1,
                current_lr,
                trainer_config.batch_size,
                &mut rng,
                AzTrainLossWeights {
                    value: trainer_config.train_value_weight,
                    policy: trainer_config.train_policy_weight,
                },
            )
            .unwrap_or_else(|err| panic!("training update {} failed: {err}", train_update));
            let train_seconds = train_started.elapsed().as_secs_f32();
            let current_lr = trainer_model
                .last_training_learning_rate()
                .unwrap_or(current_lr);
            if trainer_config.checkpoint_interval > 0
                && train_update.is_multiple_of(trainer_config.checkpoint_interval)
            {
                let path = save_checkpoint_model(
                    &trainer_model,
                    &trainer_config.model_path,
                    &trainer_config.checkpoint_dir,
                    train_update,
                );
                trainer_model
                    .save_training_state(
                        optimizer_checkpoint_path(&path),
                        train_update.saturating_add(1),
                    )
                    .unwrap_or_else(|err| {
                        panic!("failed to save checkpoint SGD state: {err}")
                    });
            }
            let mut report = build_async_training_report(
                pending,
                pending_games,
                stats,
                current_lr,
                train_data_len,
                train_seconds,
                pool.sample_count(),
                pool.capacity(),
                pool.window_stats(trainer_config.replay_recent_games),
                target_entropy,
            );
            report.training_steps = trainer_model.training_steps();
            report.training_chunks = training_chunks;
            report.test_chunks = test_chunks;
            report.holdout_checks = trainer_model.take_training_checks();
            report.cycle_complete = report.training_steps == cycle_end;
            if report.cycle_complete {
                save_model(&trainer_model, Path::new(&trainer_config.model_path));
                trainer_model.save_training_state(
                    &optimizer_state_path,
                    train_update.saturating_add(1),
                )?;
                pool.save_snapshot_lz4(&trainer_snapshot_path)?;
                cycle_end += chineseai::az::PX0_CYCLE_STEPS;
            }
            let candidate_model = trainer_model.clone();
            publish_selfplay_model(
                &trainer_shared_model,
                Arc::new(candidate_model.clone()),
                train_update,
            );
            if trainer_tx
                .send(TrainerEvent {
                    report,
                    candidate_model,
                })
                .is_err()
            {
                break 'training;
            }
            train_index += 1;
        }
        if train_index > 0 {
            trainer_model.save_training_state(
                &optimizer_state_path,
                trainer_start_update.saturating_add(train_index),
            )?;
        }
        if let Some(pool) = trainer_pool.as_mut()
            && trainer_stop.load(Ordering::SeqCst)
        {
            pool.save_snapshot_lz4(&trainer_snapshot_path)?;
        }
        Ok(())
    });
    let mut exited_after_ctrl_c = false;
    let mut exited_after_target_update = false;
    let mut update = start_update;
    let mut interrupt_save_model: Option<AzNnue> = None;
    let mut interrupt_save_next_update = start_update;
    loop {
        if interrupted.load(Ordering::SeqCst) {
            exited_after_ctrl_c = true;
            break;
        }
        let (report, candidate_model) = loop {
            match trainer_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(TrainerEvent {
                    report,
                    candidate_model,
                }) => break (report, candidate_model),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if interrupted.load(Ordering::SeqCst) {
                        exited_after_ctrl_c = true;
                        break (
                            AzLoopReport {
                                games: 0,
                                samples: 0,
                                red_wins: 0,
                                black_wins: 0,
                                draws: 0,
                                avg_plies: 0.0,
                                loss: 0.0,
                                learning_rate: 0.0,
                                value_loss: 0.0,
                                value_mse: 0.0,
                                value_pred_mean: 0.0,
                                value_target_mean: 0.0,
                                value_pred_rms: 0.0,
                                value_target_rms: 0.0,
                                value_corr: 0.0,
                                value_calibration: 0.0,
                                policy_ce: 0.0,
                                policy_kl: 0.0,
                                root_visit_entropy: 0.0,
                                entropy_opening: 0.0,
                                entropy_mid: 0.0,
                                raw_prior_top1: 0.0,
                                raw_prior_top2: 0.0,
                                policy_top1: 0.0,
                                policy_top2: 0.0,
                                root_q_gap: 0.0,
                                root_q_top1_abs: 0.0,
                                visited_actions: 0.0,
                                opening_raw_prior_top1: 0.0,
                                opening_raw_prior_top2: 0.0,
                                opening_policy_top1: 0.0,
                                opening_policy_top2: 0.0,
                                opening_q_gap: 0.0,
                                opening_q_top1_abs: 0.0,
                                opening_visited_actions: 0.0,
                                sampled_best_rate: 0.0,
                                avg_best_played_q_gap: 0.0,
                                avg_played_top_visit_ratio: 0.0,
                                avg_best_q: 0.0,
                                avg_played_q: 0.0,
                                train_seconds: 0.0,
                                total_seconds: 0.0,
                                games_per_second: 0.0,
                                samples_per_second: 0.0,
                                train_samples_per_second: 0.0,
                                train_samples: 0,
                                pool_samples: 0,
                                pool_capacity: config.replay_capacity,
                                terminal_no_legal_moves: 0,
                                terminal_red_general_missing: 0,
                                terminal_black_general_missing: 0,
                                terminal_rule_draw: 0,
                                terminal_rule_draw_natural_limit: 0,
                                terminal_rule_draw_insufficient_material: 0,
                                terminal_rule_draw_repetition: 0,
                                terminal_rule_draw_mutual_long_check: 0,
                                terminal_rule_draw_mutual_long_chase: 0,
                                terminal_rule_win_red: 0,
                                terminal_rule_win_black: 0,
                                terminal_max_plies: 0,
                                ..AzLoopReport::default()
                            },
                            AzNnue::random_with_arch(config.arch(), config.seed),
                        );
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    if interrupted.load(Ordering::SeqCst) {
                        exited_after_ctrl_c = true;
                        break (
                            AzLoopReport {
                                games: 0,
                                samples: 0,
                                red_wins: 0,
                                black_wins: 0,
                                draws: 0,
                                avg_plies: 0.0,
                                loss: 0.0,
                                learning_rate: 0.0,
                                value_loss: 0.0,
                                value_mse: 0.0,
                                value_pred_mean: 0.0,
                                value_target_mean: 0.0,
                                value_pred_rms: 0.0,
                                value_target_rms: 0.0,
                                value_corr: 0.0,
                                value_calibration: 0.0,
                                policy_ce: 0.0,
                                policy_kl: 0.0,
                                root_visit_entropy: 0.0,
                                entropy_opening: 0.0,
                                entropy_mid: 0.0,
                                raw_prior_top1: 0.0,
                                raw_prior_top2: 0.0,
                                policy_top1: 0.0,
                                policy_top2: 0.0,
                                root_q_gap: 0.0,
                                root_q_top1_abs: 0.0,
                                visited_actions: 0.0,
                                opening_raw_prior_top1: 0.0,
                                opening_raw_prior_top2: 0.0,
                                opening_policy_top1: 0.0,
                                opening_policy_top2: 0.0,
                                opening_q_gap: 0.0,
                                opening_q_top1_abs: 0.0,
                                opening_visited_actions: 0.0,
                                sampled_best_rate: 0.0,
                                avg_best_played_q_gap: 0.0,
                                avg_played_top_visit_ratio: 0.0,
                                avg_best_q: 0.0,
                                avg_played_q: 0.0,
                                train_seconds: 0.0,
                                total_seconds: 0.0,
                                games_per_second: 0.0,
                                samples_per_second: 0.0,
                                train_samples_per_second: 0.0,
                                train_samples: 0,
                                pool_samples: 0,
                                pool_capacity: config.replay_capacity,
                                terminal_no_legal_moves: 0,
                                terminal_red_general_missing: 0,
                                terminal_black_general_missing: 0,
                                terminal_rule_draw: 0,
                                terminal_rule_draw_natural_limit: 0,
                                terminal_rule_draw_insufficient_material: 0,
                                terminal_rule_draw_repetition: 0,
                                terminal_rule_draw_mutual_long_check: 0,
                                terminal_rule_draw_mutual_long_chase: 0,
                                terminal_rule_win_red: 0,
                                terminal_rule_win_black: 0,
                                terminal_max_plies: 0,
                                ..AzLoopReport::default()
                            },
                            AzNnue::random_with_arch(config.arch(), config.seed),
                        );
                    }
                    panic!("training thread exited before update {update}");
                }
            }
        };
        if exited_after_ctrl_c {
            break;
        }
        generated_games_total = generated_games_total.saturating_add(report.games as u64);
        generated_samples_total =
            generated_samples_total.saturating_add(report.samples as u64);
        let deployed_model = candidate_model.clone();
        interrupt_save_model = Some(candidate_model.clone());
        interrupt_save_next_update = update.saturating_add(1);
        let checkpoint_saved = if config.checkpoint_interval > 0
            && update.is_multiple_of(config.checkpoint_interval)
        {
            let path = checkpoint_path(&config.model_path, &config.checkpoint_dir, update);
            prune_old_checkpoints(
                &config.model_path,
                &config.checkpoint_dir,
                config.max_checkpoints,
            )
            .unwrap_or_else(|err| {
                panic!(
                    "failed to prune checkpoints in `{}`: {err}",
                    config.checkpoint_dir
                );
            });
            Some(path)
        } else {
            None
        };
        let value_rmse = report.value_mse.max(0.0).sqrt();
        let truncated = report.terminal_max_plies + report.terminal_search_no_move;
        let completed = report.games.saturating_sub(truncated);
        let true_draws = report.draws.saturating_sub(truncated);
        console.update(
            update,
            &report,
            generated_games_total,
            true_draws,
            checkpoint_saved.is_some(),
        );
        for check in &report.holdout_checks {
            console.test(check);
            for (tag, value) in [
                ("test/loss", check.loss),
                ("test/policy_kl", check.policy_kl),
                ("test/samples", check.samples as f32),
                ("test/value_samples", check.value_samples as f32),
            ] {
                log_scalar(&mut tb, tag, check.step, value);
            }
            if check.value_samples > 0 {
                log_scalar(&mut tb, "test/wdl_ce", check.step, check.value_loss);
                log_scalar(&mut tb, "test/value_rmse", check.step, check.value_rmse);
            }
        }
        for (tag, value) in [
            ("train/optimized_loss", report.loss),
            ("train/wdl_ce", report.value_loss),
            ("train/policy_kl", report.policy_kl),
            ("train/value_rmse", value_rmse),
            ("train/value_corr", report.value_corr),
            ("train/value_calibration", report.value_calibration),
            ("train/learning_rate", report.learning_rate),
            ("train/samples", report.train_samples as f32),
            (
                "train/value_samples",
                report
                    .phase_value
                    .iter()
                    .map(|phase| phase.samples)
                    .sum::<usize>() as f32,
            ),
            ("train/seconds", report.train_seconds),
            ("replay/samples", report.pool_samples as f32),
            ("selfplay/games_total", generated_games_total as f32),
            ("selfplay/samples_total", generated_samples_total as f32),
            (
                "selfplay/avg_search_simulations",
                report.avg_search_simulations,
            ),
            ("selfplay/avg_plies", report.avg_plies),
            ("selfplay/completed_games", completed as f32),
            ("selfplay/visit_policy_entropy", report.root_visit_entropy),
            (
                "truncation/rate",
                truncated as f32 / report.games.max(1) as f32,
            ),
            ("truncation/max_plies", report.terminal_max_plies as f32),
            (
                "truncation/search_no_move",
                report.terminal_search_no_move as f32,
            ),
            ("terminal/checkmate", report.terminal_checkmate as f32),
            ("terminal/stalemate", report.terminal_stalemate as f32),
            ("terminal/rule_blocked", report.terminal_rule_blocked as f32),
            (
                "terminal/red_general_missing",
                report.terminal_red_general_missing as f32,
            ),
            (
                "terminal/black_general_missing",
                report.terminal_black_general_missing as f32,
            ),
            ("terminal/rule_draw", report.terminal_rule_draw as f32),
            (
                "terminal/draw_natural_limit",
                report.terminal_rule_draw_natural_limit as f32,
            ),
            (
                "terminal/draw_insufficient_material",
                report.terminal_rule_draw_insufficient_material as f32,
            ),
            (
                "terminal/draw_repetition",
                report.terminal_rule_draw_repetition as f32,
            ),
            (
                "terminal/draw_mutual_long_check",
                report.terminal_rule_draw_mutual_long_check as f32,
            ),
            (
                "terminal/draw_mutual_long_chase",
                report.terminal_rule_draw_mutual_long_chase as f32,
            ),
            ("terminal/rule_win_red", report.terminal_rule_win_red as f32),
            (
                "terminal/rule_win_black",
                report.terminal_rule_win_black as f32,
            ),
            (
                "terminal/search_proven",
                report.terminal_search_proven.iter().sum::<usize>() as f32,
            ),
        ] {
            log_scalar(&mut tb, tag, report.training_steps, value);
        }
        if completed > 0 {
            log_scalar(
                &mut tb,
                "selfplay/draw_rate_completed",
                update,
                true_draws as f32 / completed as f32,
            );
        }
        if config.arena_interval > 0 && update.is_multiple_of(config.arena_interval) {
            {
                let (mut arena_start_positions, _arena_mode) =
                    build_arena_start_positions(&config, update);
                shuffle_positions(
                    &mut arena_start_positions,
                    &mut SplitMix64::new(
                        config.seed ^ (update as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB),
                    ),
                );
                let arena_position_count = arena_start_positions.len();
                let previous_index = champion_paths.len().checked_sub(2);
                let gate_index = update / config.arena_interval.max(1);
                let nemesis_index = arena_nemesis_update.and_then(|nemesis_update| {
                    champion_paths
                        .iter()
                        .position(|path| checkpoint_number(path) == Some(nemesis_update))
                });
                let anchor_index = nemesis_index
                    .or_else(|| historical_anchor_index(champion_paths.len(), gate_index));
                let (current_count, previous_count, _) = arena_gate_position_counts(
                    arena_position_count,
                    previous_index.is_some(),
                    anchor_index.is_some(),
                );
                let anchor_positions = arena_start_positions
                    .split_off(current_count.saturating_add(previous_count));
                let previous_positions = arena_start_positions.split_off(current_count);
                let current_positions = arena_start_positions;
                let candidate = Arc::new(deployed_model.clone());
                let run_gate_match =
                    |baseline: Arc<AzNnue>, positions: Vec<Position>, seed_salt: u64| {
                        run_arena_threads(ArenaThreadConfig {
                            candidate: Arc::clone(&candidate),
                            baseline,
                            eval_starts: ArenaStarts::Positions(Arc::new(positions)),
                            simulations: config.arena_simulations,
                            max_plies: config.max_plies,
                            rule60_max_ply: config
                                .sixty_move_rule
                                .then_some(config.rule60_max_ply),
                            cpuct: config.arena_cpuct,
                            cpuct_at_root: config.arena_cpuct_at_root,
                            cpuct_base: config.cpuct_base,
                            cpuct_factor: config.cpuct_factor,
                            cpuct_base_at_root: config.cpuct_base_at_root,
                            cpuct_factor_at_root: config.cpuct_factor_at_root,
                            fpu_value: 0.23,
                            fpu_value_at_root: 1.0,
                            draw_score: config.draw_score,
                            policy_softmax_temp: config.arena_policy_softmax_temp,
                            thread_count: config.arena_processes,
                            seed: config.seed
                                ^ (update as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                ^ seed_salt,
                        })
                    };
                let current_arena = run_gate_match(
                    Arc::new(arena_reference_model.clone()),
                    current_positions,
                    0,
                );
                let load_champion = |index: usize| {
                    let path = &champion_paths[index];
                    let model = AzNnue::load(path).unwrap_or_else(|err| {
                        panic!("failed to load champion `{}`: {err}", path.display())
                    });
                    assert_eq!(
                        model.arch,
                        deployed_model.arch,
                        "champion `{}` architecture mismatch",
                        path.display()
                    );
                    Arc::new(model)
                };
                let previous_arena = previous_index.map(|index| {
                    run_gate_match(
                        load_champion(index),
                        previous_positions,
                        0xA076_1D64_78BD_642F,
                    )
                });
                let anchor_arena = anchor_index.map(|index| {
                    run_gate_match(
                        load_champion(index),
                        anchor_positions,
                        0xE703_7ED1_A0B4_28DB,
                    )
                });
                let elo_diff = current_arena.elo_diff_vs_even();
                let (elo_lower, elo_upper) =
                    current_arena.elo_diff_bounds(config.arena_promotion_confidence_z);
                let gate_decision = arena_gate_decision(
                    &current_arena,
                    previous_arena.as_ref(),
                    anchor_arena.as_ref(),
                    config.arena_promotion_rate,
                    config.arena_promotion_confidence_z,
                );
                let promoted = gate_decision == ArenaGateDecision::Promote;
                if let (Some(index), Some(report)) = (anchor_index, anchor_arena.as_ref()) {
                    if report.score_rate_upper_bound(config.arena_promotion_confidence_z)
                        < 0.50
                    {
                        arena_nemesis_update = checkpoint_number(&champion_paths[index]);
                    } else if promoted && nemesis_index == Some(index) {
                        arena_nemesis_update = None;
                    }
                }
                if promoted {
                    arena_reference_model = deployed_model.clone();
                    let best_checkpoint = save_best_checkpoint_model(
                        &deployed_model,
                        &config.model_path,
                        &config.checkpoint_dir,
                        update,
                    );
                    save_model(&deployed_model, &best_path);
                    champion_paths.push(best_checkpoint.clone());
                }

                console.arena(format!(
                    "arena {update:04}: games={} W/L/D={}/{}/{} score={:.3} ci={:.3}..{:.3} previous={} anchor={} decision={:?}",
                    current_arena.total_games(),
                    current_arena.wins,
                    current_arena.losses,
                    current_arena.draws,
                    current_arena.score_rate(),
                    current_arena
                        .score_rate_lower_bound(config.arena_promotion_confidence_z),
                    current_arena
                        .score_rate_upper_bound(config.arena_promotion_confidence_z),
                    previous_arena
                        .as_ref()
                        .map_or_else(|| "-".into(), |r| format!("{:.3}", r.score_rate())),
                    anchor_arena
                        .as_ref()
                        .map_or_else(|| "-".into(), |r| format!("{:.3}", r.score_rate())),
                    gate_decision
                ));
                let mut historical_arena = AzArenaReport::default();
                if let Some(report) = previous_arena.as_ref() {
                    historical_arena.add_assign(report);
                }
                if let Some(report) = anchor_arena.as_ref() {
                    historical_arena.add_assign(report);
                }
                if historical_arena.total_games() > 0 {
                    log_scalar(
                        &mut tb,
                        "arena/history_score_rate",
                        update,
                        historical_arena.score_rate(),
                    );
                }
                log_scalar(
                    &mut tb,
                    "arena/score_rate",
                    update,
                    current_arena.score_rate(),
                );
                if let Some(report) = previous_arena.as_ref() {
                    log_scalar(
                        &mut tb,
                        "arena/previous_score_rate",
                        update,
                        report.score_rate(),
                    );
                }
                if let Some(report) = anchor_arena.as_ref() {
                    log_scalar(
                        &mut tb,
                        "arena/anchor_score_rate",
                        update,
                        report.score_rate(),
                    );
                }
                log_scalar(&mut tb, "arena/elo_diff", update, elo_diff);
                log_scalar(&mut tb, "arena/elo_diff_lower", update, elo_lower);
                log_scalar(&mut tb, "arena/elo_diff_upper", update, elo_upper);
                log_scalar(
                    &mut tb,
                    "arena/wins_as_red",
                    update,
                    current_arena.wins_as_red as f32,
                );
                log_scalar(
                    &mut tb,
                    "arena/losses_as_red",
                    update,
                    current_arena.losses_as_red as f32,
                );
                log_scalar(
                    &mut tb,
                    "arena/wins_as_black",
                    update,
                    current_arena.wins_as_black as f32,
                );
                log_scalar(
                    &mut tb,
                    "arena/losses_as_black",
                    update,
                    current_arena.losses_as_black as f32,
                );
                log_scalar(
                    &mut tb,
                    "arena/promoted",
                    update,
                    if promoted { 1.0 } else { 0.0 },
                );
            }
        }
        if config.pikafish_label_eval_interval > 0
            && update.is_multiple_of(config.pikafish_label_eval_interval)
            && !config.pikafish_label_eval_sqlite.trim().is_empty()
        {
            let sqlite_path = Path::new(&config.pikafish_label_eval_sqlite);
            if sqlite_path.exists() {
                let started = Instant::now();
                let eval_result = (|| -> io::Result<LabelEvalStats> {
                    let conn = Connection::open(sqlite_path).map_err(sqlite_io_error)?;
                    let rows = load_pikafish_label_rows(
                        &conn,
                        config.pikafish_label_eval_limit,
                        config.seed,
                    )
                    .map_err(sqlite_io_error)?;
                    evaluate_pikafish_labels_parallel(
                        Arc::new(deployed_model.clone()),
                        rows,
                        AzSearchLimits {
                            simulations: config.pikafish_label_eval_simulations,
                            seed: config.seed
                                ^ (update as u64).wrapping_mul(0xD6E8_FD50_19B7_8421),
                            cpuct: config.pikafish_label_eval_cpuct,
                            cpuct_at_root: config.pikafish_label_eval_cpuct_at_root,
                            cpuct_base: config.cpuct_base,
                            cpuct_factor: config.cpuct_factor,
                            cpuct_base_at_root: config.cpuct_base_at_root,
                            cpuct_factor_at_root: config.cpuct_factor_at_root,
                            max_depth: config.max_plies,
                            root_dirichlet_alpha: 0.0,
                            root_exploration_fraction: 0.0,
                            fpu_value: 0.23,
                            fpu_value_at_root: 1.0,
                            fpu_absolute_at_root: true,
                            minimum_kldgain_per_node: 0.0,
                            policy_softmax_temp: config
                                .pikafish_label_eval_policy_softmax_temp,
                            draw_score: config.draw_score,
                            value_scale: 1.0,
                        },
                        config.arena_processes,
                    )
                })();
                match eval_result {
                    Ok(stats) => {
                        console.event(format!(
                            "pikafish-label {update:04}: sqlite={} evaluated={} legal={} value_labels={} sims={} threads={} search_top1={:.3}% search_top2={:.3}% search_top4={:.3}% search_top8={:.3}% raw_prior_top1={:.3}% raw_value_corr={:.4} raw_value_mae={:.4} search_value_corr={:.4} search_value_mae={:.4} elapsed={:.1}s",
                            config.pikafish_label_eval_sqlite,
                            stats.count,
                            stats.legal_bestmove,
                            stats.value_count(),
                            config.pikafish_label_eval_simulations,
                            config.arena_processes,
                            100.0 * stats.top1_rate(),
                            100.0 * stats.top2_rate(),
                            100.0 * stats.top4_rate(),
                            100.0 * stats.top8_rate(),
                            100.0 * stats.prior_top1_rate(),
                            stats.raw_value_corr(),
                            stats.raw_value_mae_wdl_q(),
                            stats.value_corr(),
                            stats.value_mae_wdl_q(),
                            started.elapsed().as_secs_f32()
                        ));
                        log_scalar(
                            &mut tb,
                            "pikafish_label/evaluated_positions",
                            update,
                            stats.count as f32,
                        );
                        log_scalar(
                            &mut tb,
                            "pikafish_label/search_top1",
                            update,
                            stats.top1_rate(),
                        );
                        log_scalar(
                            &mut tb,
                            "pikafish_label/search_top2",
                            update,
                            stats.top2_rate(),
                        );
                        log_scalar(
                            &mut tb,
                            "pikafish_label/search_top4",
                            update,
                            stats.top4_rate(),
                        );
                        log_scalar(
                            &mut tb,
                            "pikafish_label/search_top8",
                            update,
                            stats.top8_rate(),
                        );
                        log_scalar(
                            &mut tb,
                            "pikafish_label/raw_prior_top1",
                            update,
                            stats.prior_top1_rate(),
                        );
                        log_scalar(
                            &mut tb,
                            "pikafish_label/value_labels",
                            update,
                            stats.value_count() as f32,
                        );
                        log_scalar(
                            &mut tb,
                            "pikafish_label/search_value_corr",
                            update,
                            stats.value_corr() as f32,
                        );
                        log_scalar(
                            &mut tb,
                            "pikafish_label/search_value_mae_wdl_q",
                            update,
                            stats.value_mae_wdl_q(),
                        );
                        log_scalar(
                            &mut tb,
                            "pikafish_label/raw_value_corr",
                            update,
                            stats.raw_value_corr() as f32,
                        );
                        log_scalar(
                            &mut tb,
                            "pikafish_label/raw_value_mae_wdl_q",
                            update,
                            stats.raw_value_mae_wdl_q(),
                        );
                    }
                    Err(err) => {
                        console.event(format!(
                            "pikafish-label {update:04}: failed sqlite={}: {err}",
                            config.pikafish_label_eval_sqlite
                        ));
                    }
                }
            } else {
                let resolved = if sqlite_path.is_absolute() {
                    sqlite_path.to_path_buf()
                } else {
                    std::env::current_dir()
                        .unwrap_or_else(|_| PathBuf::from("."))
                        .join(sqlite_path)
                };
                console.event(format!(
                    "pikafish-label {update:04}: skipped missing sqlite={} resolved={} (copy the label DB or update pikafish_label_eval_sqlite)",
                    config.pikafish_label_eval_sqlite,
                    resolved.display()
                ));
            }
        }
        tb.flush();
        update = update.saturating_add(1);
        if report.cycle_complete {
            save_az_loop_progress_pair(
                &config_path,
                interrupt_save_next_update,
                arena_nemesis_update,
                generated_games_total,
                generated_samples_total,
            );
            console.event(format!(
                "saved: cycle {} complete; optimizer+replay saved; continuing cycle {} next_update={}",
                report.training_steps / chineseai::az::PX0_CYCLE_STEPS,
                report.training_steps / chineseai::az::PX0_CYCLE_STEPS + 1,
                interrupt_save_next_update,
            ));
        }
        if let Some(target_update) = target_update
            && update > target_update
        {
            exited_after_target_update = true;
            break;
        }
    }
    console.finish();
    stop_requested.store(true, Ordering::SeqCst);
    // 等待线程前持续排空结果队列，避免满队列让训练及产数线程相互等待。
    for event in trainer_rx {
        if exited_after_ctrl_c {
            generated_games_total =
                generated_games_total.saturating_add(event.report.games as u64);
            generated_samples_total =
                generated_samples_total.saturating_add(event.report.samples as u64);
            interrupt_save_model = Some(event.candidate_model);
            interrupt_save_next_update = update.saturating_add(1);
            update = update.saturating_add(1);
        }
    }
    for handle in selfplay_handles {
        handle
            .join()
            .unwrap_or_else(|_| panic!("selfplay thread panicked"));
    }
    collector_handle
        .join()
        .unwrap_or_else(|_| panic!("selfplay collector thread panicked"));
    trainer_handle
        .join()
        .unwrap_or_else(|_| panic!("training thread panicked"))
        .unwrap_or_else(|err| panic!("failed to save training state: {err}"));
    if exited_after_ctrl_c || exited_after_target_update {
        if let Some(model) = interrupt_save_model.as_ref() {
            save_model(model, Path::new(&config.model_path));
            save_az_loop_progress_pair(
                &config_path,
                interrupt_save_next_update,
                arena_nemesis_update,
                generated_games_total,
                generated_samples_total,
            );
            println!(
                "saved: {} model=`{}` optimizer+replay saved next_update={}",
                if exited_after_target_update {
                    "target"
                } else {
                    "interrupt"
                },
                config.model_path,
                interrupt_save_next_update
            );
        } else {
            println!(
                "model    : no completed update to save on {}",
                if exited_after_target_update {
                    "target stop"
                } else {
                    "interrupt"
                }
            );
        }
    }
    true
}

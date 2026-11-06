//! 自动发现本地资源、预检后进入训练；已有配置和训练状态保持原位。
use crate::cli::{args::AzLoopArgs, az_loop, az_loop_config::AzLoopFileConfig};
use chineseai::{
    pikafish::{
        opening_book::Px0OpeningBook,
        teacher::{PikafishTeacher, TeacherBudget, TeacherConfig},
    },
    xiangqi::Position,
};
use clap::Args;
use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Args, Debug)]
pub(crate) struct AzTrainArgs {
    /// 训练配置；不存在时自动建立本地教师训练配置。
    #[arg(default_value = "training/local-endgame.toml")]
    pub(crate) config: String,
    /// 完成这个绝对更新编号后保存退出。
    #[arg(long)]
    pub(crate) target_update: Option<usize>,
    /// 只检查资源并准备配置，不创建模型或启动训练。
    #[arg(long)]
    pub(crate) prepare_only: bool,
}

pub(crate) fn run(cmd: AzTrainArgs) -> bool {
    let (config_path, config) = prepare_training(Path::new(&cmd.config))
        .unwrap_or_else(|error| fail(format!("训练准备失败：{error}")));
    println!("train-ready: config={}", config_path.display());
    println!("model      : {}", config.model_path);
    println!(
        "curriculum : restart={} teacher={} deblunder={}",
        config.restart_fraction, config.endgame_teacher_enabled, config.deblunder_enabled
    );
    if cmd.prepare_only {
        return false;
    }
    let model_parent = Path::new(&config.model_path).parent().unwrap();
    fs::create_dir_all(model_parent)
        .unwrap_or_else(|error| fail(format!("无法创建模型目录：{error}")));
    az_loop::run_config(
        AzLoopArgs {
            config: path_string(&config_path),
            target_update: cmd.target_update,
        },
        config,
    )
}

fn fail(message: String) -> ! {
    eprintln!("{message}");
    std::process::exit(1)
}

fn prepare_training(config_path: &Path) -> io::Result<(PathBuf, AzLoopFileConfig)> {
    let config_path = std::path::absolute(config_path)?;
    let config_dir = config_path.parent().unwrap();
    let project_root = discover_project_root(&config_path)?;
    let exists = config_path.try_exists()?;
    let mut config = if exists {
        AzLoopFileConfig::parse(&fs::read_to_string(&config_path)?)
    } else {
        ensure_new_run_is_empty(&config_path)?;
        new_preset(&project_root, &config_path)?
    };
    resolve_config_paths(&mut config, config_dir, &project_root)?;
    validate_resources(&config)?;
    if !exists {
        fs::create_dir_all(config_dir)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&config_path)?;
        file.write_all(config.to_file_text().as_bytes())?;
        println!("created    : {}", config_path.display());
    }
    Ok((config_path, config))
}

fn discover_project_root(config_path: &Path) -> io::Result<PathBuf> {
    let current = std::env::current_dir()?;
    let executable = std::env::current_exe()?;
    for start in [
        current.as_path(),
        config_path.parent().unwrap(),
        executable.parent().unwrap(),
    ] {
        if let Some(root) = start
            .ancestors()
            .find(|root| root.join("Cargo.toml").is_file() && root.join("src").is_dir())
        {
            return Ok(root.to_path_buf());
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "未找到项目根目录；请从项目内运行 az-train，或使用 tools/train.ps1",
    ))
}

fn run_directory(config_path: &Path) -> io::Result<PathBuf> {
    let stem = config_path
        .file_stem()
        .filter(|stem| !stem.is_empty())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "训练配置需要文件名"))?;
    Ok(config_path.parent().unwrap().join(stem))
}

fn ensure_new_run_is_empty(config_path: &Path) -> io::Result<()> {
    let run_dir = run_directory(config_path)?;
    let occupied = run_dir.try_exists()?
        || [".progress", ".replay.lz4", ".adamw.safetensors"]
            .iter()
            .any(|suffix| Path::new(&format!("{}{suffix}", config_path.display())).exists());
    if occupied {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "配置缺失但对应训练目录或进度已存在；请恢复原配置，或选择新的配置文件名",
        ));
    }
    Ok(())
}

fn new_preset(project_root: &Path, config_path: &Path) -> io::Result<AzLoopFileConfig> {
    let run_dir = run_directory(config_path)?;
    let physical_cores = num_cpus::get_physical().max(1);
    let teacher_processes = physical_cores.min(2);
    let workers = physical_cores.saturating_sub(teacher_processes).clamp(1, 8);
    let eval_database = project_root.join("eval/pikafish-selfplay-5000-d20.sqlite");
    let mut config = AzLoopFileConfig {
        model_path: path_string(&run_dir.join("model.safetensors")),
        checkpoint_dir: path_string(&run_dir.join("checkpoints")),
        tensorboard_logdir: path_string(&run_dir.join("logs")),
        selfplay_opening_book: path_string(&project_root.join("book.pgn.gz")),
        arena_opening_book: path_string(&project_root.join("book.pgn.gz")),
        restart_fraction: 0.5,
        // 教师完成对局后使用其真实结果，避免弱模型 Q 覆盖这些结果。
        deblunder_enabled: false,
        endgame_teacher_enabled: true,
        endgame_teacher_exe: path_string(&project_root.join("tools/pikafish.exe")),
        endgame_teacher_nnue: path_string(&project_root.join("tools/pikafish.nnue")),
        endgame_teacher_processes: teacher_processes,
        workers,
        arena_processes: workers,
        ..Default::default()
    };
    if eval_database.is_file() {
        config.pikafish_label_eval_sqlite = path_string(&eval_database);
    } else {
        config.pikafish_label_eval_sqlite.clear();
        config.pikafish_label_eval_interval = 0;
    }
    Ok(config)
}

fn resolve_config_paths(
    config: &mut AzLoopFileConfig,
    config_dir: &Path,
    project_root: &Path,
) -> io::Result<()> {
    for value in [
        &mut config.model_path,
        &mut config.checkpoint_dir,
        &mut config.tensorboard_logdir,
    ] {
        if !value.trim().is_empty() && Path::new(value.as_str()).is_relative() {
            *value = path_string(&config_dir.join(&*value));
        }
    }
    config.selfplay_opening_book =
        resolve_input(&config.selfplay_opening_book, config_dir, project_root)?;
    if config.arena_interval > 0 {
        config.arena_opening_book =
            resolve_input(&config.arena_opening_book, config_dir, project_root)?;
    }
    if config.pikafish_label_eval_interval > 0
        && !config.pikafish_label_eval_sqlite.trim().is_empty()
    {
        config.pikafish_label_eval_sqlite =
            resolve_input(&config.pikafish_label_eval_sqlite, config_dir, project_root)?;
    }
    if !config.tablebase_sqlite.trim().is_empty() {
        config.tablebase_sqlite =
            resolve_input(&config.tablebase_sqlite, config_dir, project_root)?;
    }
    if config.endgame_teacher_enabled {
        config.endgame_teacher_exe =
            resolve_input(&config.endgame_teacher_exe, config_dir, project_root)?;
        config.endgame_teacher_nnue =
            resolve_input(&config.endgame_teacher_nnue, config_dir, project_root)?;
    }
    Ok(())
}

fn resolve_input(value: &str, config_dir: &Path, project_root: &Path) -> io::Result<String> {
    let configured = Path::new(value);
    if configured.is_absolute() {
        return require_file(configured);
    }
    let beside_config = config_dir.join(configured);
    if beside_config.is_file() {
        return require_file(&beside_config);
    }
    require_file(&project_root.join(configured))
}

fn require_file(path: &Path) -> io::Result<String> {
    let metadata = fs::metadata(path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("资源 `{}` 不可用：{error}", path.display()),
        )
    })?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("资源 `{}` 必须是非空文件", path.display()),
        ));
    }
    Ok(path_string(&std::path::absolute(path)?))
}

fn validate_resources(config: &AzLoopFileConfig) -> io::Result<()> {
    let mut openings = Px0OpeningBook::load(&config.selfplay_opening_book, config.seed)?;
    openings.next_batch(1, 0)?;
    println!("opening    : {} positions", openings.len());
    if config.arena_interval > 0 && config.arena_opening_book != config.selfplay_opening_book {
        Px0OpeningBook::load(&config.arena_opening_book, config.seed)?.next_batch(1, 0)?;
    }
    if !config.tablebase_sqlite.trim().is_empty() {
        let labels = chineseai::az::AzTablebase::load(&config.tablebase_sqlite)?;
        if labels.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "残局缓存没有与当前规则版本匹配的已验证标签",
            ));
        }
        println!("tablebase  : {} verified positions", labels.len());
    }
    if config.endgame_teacher_enabled {
        let mut teacher = PikafishTeacher::spawn(&TeacherConfig {
            exe: PathBuf::from(&config.endgame_teacher_exe),
            nnue: PathBuf::from(&config.endgame_teacher_nnue),
            hash_mb: config.endgame_teacher_hash_mb,
            timeout: Duration::from_millis(config.endgame_teacher_timeout_ms),
        })?;
        let position = Position::startpos();
        let legal = position.legal_moves_with_rules(&position.initial_rule_history());
        let mv = teacher.best_move("position startpos", &legal, TeacherBudget::Depth(1))?;
        println!("teacher    : NNUE loaded, legal move {}", mv.to_uci());
    }
    Ok(())
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_directory(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "chineseai-az-train-{name}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn preset_isolates_artifacts_and_enables_local_teacher_curriculum() {
        let root = temp_directory("preset");
        let config_path = root.join("training").join("experiment.toml");
        let config = new_preset(&root, &config_path).unwrap();
        assert_eq!(
            Path::new(&config.model_path),
            root.join("training")
                .join("experiment")
                .join("model.safetensors")
        );
        assert_ne!(
            Path::new(&config.model_path),
            root.join("model.safetensors")
        );
        assert!(config.endgame_teacher_enabled);
        assert!(!config.deblunder_enabled);
        assert_eq!(config.restart_fraction, 0.5);
        assert!((1..=8).contains(&config.workers));
        assert!(!Path::new(&config.model_path).exists());
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn missing_config_never_reuses_or_overwrites_existing_training_state() {
        let root = temp_directory("collision");
        let config_path = root.join("run.toml");
        let progress_path = root.join("run.toml.progress");
        fs::write(&progress_path, "existing-progress").unwrap();
        assert_eq!(
            ensure_new_run_is_empty(&config_path).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(
            fs::read_to_string(&progress_path).unwrap(),
            "existing-progress"
        );
        fs::remove_file(progress_path).unwrap();
        fs::create_dir(root.join("run")).unwrap();
        assert!(ensure_new_run_is_empty(&config_path).is_err());
        fs::remove_dir(root.join("run")).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn input_resolution_preserves_explicit_paths_and_prefers_config_directory() {
        let root = temp_directory("paths");
        let configs = root.join("configs");
        fs::create_dir(&configs).unwrap();
        fs::write(root.join("book.gz"), "root-book").unwrap();
        fs::write(configs.join("book.gz"), "config-book").unwrap();
        assert_eq!(
            PathBuf::from(resolve_input("book.gz", &configs, &root).unwrap()),
            configs.join("book.gz")
        );
        assert_eq!(
            PathBuf::from(
                resolve_input(&path_string(&root.join("book.gz")), &configs, &root).unwrap()
            ),
            root.join("book.gz")
        );
        fs::remove_file(configs.join("book.gz")).unwrap();
        assert_eq!(
            PathBuf::from(resolve_input("book.gz", &configs, &root).unwrap()),
            root.join("book.gz")
        );
        assert!(resolve_input(&path_string(&configs.join("missing.gz")), &configs, &root).is_err());
        fs::remove_file(root.join("book.gz")).unwrap();
        fs::remove_dir(configs).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn preparing_existing_config_preserves_its_bytes_and_model() {
        let root = temp_directory("existing");
        let book_path = root.join("book.pgn.gz");
        let mut compressed = flate2::write::GzEncoder::new(
            fs::File::create(&book_path).unwrap(),
            flate2::Compression::default(),
        );
        writeln!(
            compressed,
            "[FEN \"{}\"]\n{{}}",
            Position::startpos().to_fen()
        )
        .unwrap();
        compressed.finish().unwrap();
        let config_path = root.join("custom.toml");
        let config = AzLoopFileConfig {
            model_path: "existing.safetensors".into(),
            selfplay_opening_book: "book.pgn.gz".into(),
            arena_interval: 0,
            pikafish_label_eval_interval: 0,
            endgame_teacher_enabled: false,
            ..Default::default()
        };
        let original = config.to_file_text();
        fs::write(&config_path, &original).unwrap();
        fs::write(root.join("existing.safetensors"), "existing-model").unwrap();
        let (_, prepared) = prepare_training(&config_path).unwrap();
        assert_eq!(fs::read_to_string(&config_path).unwrap(), original);
        assert_eq!(
            Path::new(&prepared.model_path),
            root.join("existing.safetensors")
        );
        assert_eq!(
            fs::read_to_string(root.join("existing.safetensors")).unwrap(),
            "existing-model"
        );
        for path in [config_path, book_path, root.join("existing.safetensors")] {
            fs::remove_file(path).unwrap();
        }
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn missing_resource_fails_before_model_directory_creation() {
        let root = temp_directory("missing");
        let config_path = root.join("custom.toml");
        let config = AzLoopFileConfig {
            model_path: "new-model/model.safetensors".into(),
            selfplay_opening_book: path_string(&root.join("missing-book.pgn.gz")),
            arena_interval: 0,
            pikafish_label_eval_interval: 0,
            ..Default::default()
        };
        let original = config.to_file_text();
        fs::write(&config_path, &original).unwrap();
        assert!(prepare_training(&config_path).is_err());
        assert!(!root.join("new-model").exists());
        assert_eq!(fs::read_to_string(&config_path).unwrap(), original);
        fs::remove_file(config_path).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn empty_verified_tablebase_fails_before_model_directory_creation() {
        let root = temp_directory("empty-tablebase");
        let book_path = root.join("book.pgn.gz");
        let mut compressed = flate2::write::GzEncoder::new(
            fs::File::create(&book_path).unwrap(),
            flate2::Compression::default(),
        );
        writeln!(
            compressed,
            "[FEN \"{}\"]\n{{}}",
            Position::startpos().to_fen()
        )
        .unwrap();
        compressed.finish().unwrap();
        let tablebase_path = root.join("empty.sqlite");
        let connection = rusqlite::Connection::open(&tablebase_path).unwrap();
        connection
            .execute_batch(chineseai::az::TABLEBASE_SCHEMA)
            .unwrap();
        drop(connection);
        let config_path = root.join("custom.toml");
        let config = AzLoopFileConfig {
            model_path: "new-model/model.safetensors".into(),
            selfplay_opening_book: "book.pgn.gz".into(),
            tablebase_sqlite: "empty.sqlite".into(),
            arena_interval: 0,
            pikafish_label_eval_interval: 0,
            ..Default::default()
        };
        fs::write(&config_path, config.to_file_text()).unwrap();
        let error = prepare_training(&config_path).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("已验证标签"));
        assert!(!root.join("new-model").exists());
        for path in [config_path, tablebase_path, book_path] {
            fs::remove_file(path).unwrap();
        }
        fs::remove_dir(root).unwrap();
    }
}

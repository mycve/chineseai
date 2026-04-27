use axum::{
    Json, Router,
    extract::State,
    response::Html,
    routing::{get, post},
};
use chineseai::xiangqi::{
    BOARD_FILES, BOARD_RANKS, Color, Move, Piece, PieceKind, Position, RuleHistoryEntry,
    RuleOutcome, parse_square, square_name,
};
use serde::Serialize;
use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
};

#[derive(Clone)]
struct AppState {
    inner: Arc<Mutex<RuleUiState>>,
}

struct RuleUiState {
    position: Position,
    rule_history: Vec<RuleHistoryEntry>,
    undo_stack: Vec<(Position, Vec<RuleHistoryEntry>)>,
    selected: Option<usize>,
}

impl RuleUiState {
    fn new(position: Position) -> Self {
        Self {
            rule_history: position.initial_rule_history(),
            position,
            undo_stack: Vec::new(),
            selected: None,
        }
    }

    fn load(&mut self, position: Position) {
        self.rule_history = position.initial_rule_history();
        self.position = position;
        self.undo_stack.clear();
        self.selected = None;
    }

    fn undo(&mut self) -> Result<(), String> {
        let Some((position, rule_history)) = self.undo_stack.pop() else {
            return Err("没有可撤销的着法".to_string());
        };
        self.position = position;
        self.rule_history = rule_history;
        self.selected = None;
        Ok(())
    }

    fn play(&mut self, mv: Move) -> Result<(), String> {
        if !can_generate_moves(&self.position) {
            return Err("局面非法：有一方将帅不存在".to_string());
        }
        let legal = self.position.legal_moves_with_rules(&self.rule_history);
        if !legal.contains(&mv) {
            let raw = self.position.legal_moves().contains(&mv);
            return if raw {
                Err("走法本身合法，但被重复局面、长将或长捉规则禁止".to_string())
            } else {
                Err("非法走法".to_string())
            };
        }

        self.undo_stack
            .push((self.position.clone(), self.rule_history.clone()));
        self.rule_history
            .push(self.position.rule_history_entry_after_move(mv));
        self.position.make_move(mv);
        self.selected = None;
        Ok(())
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StateDto {
    fen: String,
    side: &'static str,
    halfmove: u16,
    red_general: bool,
    black_general: bool,
    red_check: bool,
    black_check: bool,
    legal_count: usize,
    rule_legal_count: usize,
    rule_outcome: Option<String>,
    message: Option<String>,
    selected: Option<String>,
    board: Vec<Vec<CellDto>>,
    legal_moves: Vec<MoveDto>,
    selected_moves: Vec<MoveDto>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CellDto {
    sq: String,
    piece: Option<char>,
    color: Option<&'static str>,
    kind: Option<&'static str>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MoveDto {
    uci: String,
    from: String,
    to: String,
    capture: bool,
}

pub fn run_rule_ui(initial: Position, host: &str, port: u16) {
    let addr: SocketAddr = format!("{host}:{port}").parse().unwrap_or_else(|err| {
        panic!("invalid rule-ui address {host}:{port}: {err}");
    });
    let state = AppState {
        inner: Arc::new(Mutex::new(RuleUiState::new(initial))),
    };

    let runtime = tokio::runtime::Runtime::new()
        .unwrap_or_else(|err| panic!("failed to start tokio runtime: {err}"));
    runtime.block_on(async move {
        let app = Router::new()
            .route("/", get(index))
            .route("/api/state", get(api_state))
            .route("/api/click", post(api_click))
            .route("/api/move", post(api_move))
            .route("/api/load", post(api_load))
            .route("/api/undo", post(api_undo))
            .route("/api/reset", post(api_reset))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .unwrap_or_else(|err| panic!("failed to bind rule-ui at http://{addr}: {err}"));
        println!("rule-ui: http://{addr}");
        axum::serve(listener, app)
            .await
            .unwrap_or_else(|err| panic!("rule-ui server failed: {err}"));
    });
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn api_state(State(state): State<AppState>) -> Json<StateDto> {
    let state = state.inner.lock().unwrap();
    Json(state_dto(&state, None))
}

async fn api_click(State(state): State<AppState>, body: String) -> Json<StateDto> {
    let mut state = state.inner.lock().unwrap();
    let message = parse_square(body.trim())
        .map(|square| handle_square_click(&mut state, square))
        .unwrap_or_else(|| Some("坐标无效".to_string()));
    Json(state_dto(&state, message))
}

async fn api_move(State(state): State<AppState>, body: String) -> Json<StateDto> {
    let mut state = state.inner.lock().unwrap();
    let message = Move::from_uci(body.trim())
        .map(|mv| match state.play(mv) {
            Ok(()) => format!("已走 {}", mv.to_uci()),
            Err(err) => err,
        })
        .unwrap_or_else(|| "着法格式无效".to_string());
    Json(state_dto(&state, Some(message)))
}

async fn api_load(State(state): State<AppState>, body: String) -> Json<StateDto> {
    let text = body.trim();
    let parsed = if text.is_empty() || text == "startpos" {
        Ok(Position::startpos())
    } else {
        Position::from_fen(text)
    };
    let mut state = state.inner.lock().unwrap();
    let message = match parsed {
        Ok(position) => {
            state.load(position);
            "局面已载入".to_string()
        }
        Err(err) => format!("FEN 无效：{err}"),
    };
    Json(state_dto(&state, Some(message)))
}

async fn api_undo(State(state): State<AppState>) -> Json<StateDto> {
    let mut state = state.inner.lock().unwrap();
    let message = match state.undo() {
        Ok(()) => "undone".to_string(),
        Err(err) => err,
    };
    Json(state_dto(&state, Some(message)))
}

async fn api_reset(State(state): State<AppState>) -> Json<StateDto> {
    let mut state = state.inner.lock().unwrap();
    state.load(Position::startpos());
    Json(state_dto(&state, Some("已恢复初始局面".to_string())))
}

fn handle_square_click(state: &mut RuleUiState, square: usize) -> Option<String> {
    if !can_generate_moves(&state.position) {
        return Some("局面非法：有一方将帅不存在".to_string());
    }

    if let Some(from) = state.selected {
        let mv = Move::new(from, square);
        if state
            .position
            .legal_moves_with_rules(&state.rule_history)
            .contains(&mv)
        {
            return match state.play(mv) {
                Ok(()) => Some(format!("已走 {}", mv.to_uci())),
                Err(err) => Some(err),
            };
        }
    }

    if state
        .position
        .piece_at(square)
        .is_some_and(|piece| piece.color == state.position.side_to_move())
    {
        state.selected = Some(square);
        Some(format!("已选中 {}", square_name(square)))
    } else {
        state.selected = None;
        Some("已取消选择".to_string())
    }
}

fn state_dto(state: &RuleUiState, message: Option<String>) -> StateDto {
    let position = &state.position;
    let legal = if can_generate_moves(position) {
        position.legal_moves()
    } else {
        Vec::new()
    };
    let rule_legal = if can_generate_moves(position) {
        position.legal_moves_with_rules(&state.rule_history)
    } else {
        Vec::new()
    };
    let selected_moves = state.selected.map_or_else(Vec::new, |from| {
        rule_legal
            .iter()
            .copied()
            .filter(|mv| mv.from as usize == from)
            .collect()
    });

    StateDto {
        fen: position.to_fen(),
        side: color_label(position.side_to_move()),
        halfmove: position.halfmove_clock(),
        red_general: position.has_general(Color::Red),
        black_general: position.has_general(Color::Black),
        red_check: can_generate_moves(position) && position.in_check(Color::Red),
        black_check: can_generate_moves(position) && position.in_check(Color::Black),
        legal_count: legal.len(),
        rule_legal_count: rule_legal.len(),
        rule_outcome: rule_outcome_text(position, &state.rule_history),
        message,
        selected: state.selected.map(square_name),
        board: board_dto(position),
        legal_moves: moves_dto(position, &rule_legal),
        selected_moves: moves_dto(position, &selected_moves),
    }
}

fn board_dto(position: &Position) -> Vec<Vec<CellDto>> {
    (0..BOARD_RANKS)
        .map(|rank| {
            (0..BOARD_FILES)
                .map(|file| {
                    let sq = rank * BOARD_FILES + file;
                    let piece = position.piece_at(sq);
                    CellDto {
                        sq: square_name(sq),
                        piece: piece.map(piece_label),
                        color: piece.map(|piece| color_class(piece.color)),
                        kind: piece.map(|piece| piece_kind_name(piece.kind)),
                    }
                })
                .collect()
        })
        .collect()
}

fn moves_dto(position: &Position, moves: &[Move]) -> Vec<MoveDto> {
    moves
        .iter()
        .map(|mv| MoveDto {
            uci: mv.to_uci(),
            from: square_name(mv.from as usize),
            to: square_name(mv.to as usize),
            capture: position.is_capture(*mv),
        })
        .collect()
}

fn rule_outcome_text(position: &Position, history: &[RuleHistoryEntry]) -> Option<String> {
    position
        .rule_outcome_with_history(history)
        .map(|outcome| match outcome {
            RuleOutcome::Draw(reason) => format!("和棋（{reason:?}）"),
            RuleOutcome::Win(color) => format!("{}胜", color_label(color)),
        })
}

fn can_generate_moves(position: &Position) -> bool {
    position.has_general(Color::Red) && position.has_general(Color::Black)
}

fn color_label(color: Color) -> &'static str {
    match color {
        Color::Red => "红方",
        Color::Black => "黑方",
    }
}

fn color_class(color: Color) -> &'static str {
    match color {
        Color::Red => "red",
        Color::Black => "black",
    }
}

fn piece_kind_name(kind: PieceKind) -> &'static str {
    match kind {
        PieceKind::General => "将帅",
        PieceKind::Advisor => "士仕",
        PieceKind::Elephant => "象相",
        PieceKind::Horse => "马",
        PieceKind::Rook => "车",
        PieceKind::Cannon => "炮",
        PieceKind::Soldier => "兵卒",
    }
}

fn piece_label(piece: Piece) -> char {
    match (piece.color, piece.kind) {
        (Color::Red, PieceKind::General) => '帥',
        (Color::Red, PieceKind::Advisor) => '仕',
        (Color::Red, PieceKind::Elephant) => '相',
        (Color::Red, PieceKind::Horse) => '馬',
        (Color::Red, PieceKind::Rook) => '車',
        (Color::Red, PieceKind::Cannon) => '炮',
        (Color::Red, PieceKind::Soldier) => '兵',
        (Color::Black, PieceKind::General) => '將',
        (Color::Black, PieceKind::Advisor) => '士',
        (Color::Black, PieceKind::Elephant) => '象',
        (Color::Black, PieceKind::Horse) => '馬',
        (Color::Black, PieceKind::Rook) => '車',
        (Color::Black, PieceKind::Cannon) => '砲',
        (Color::Black, PieceKind::Soldier) => '卒',
    }
}

const INDEX_HTML: &str = r#"<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChineseAI 规则棋盘</title>
<script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
<style>
:root{font-family:"Microsoft YaHei","PingFang SC","Noto Sans SC",Segoe UI,Arial,sans-serif;color:#24180f;background:#eee6d6}
body{margin:0;background:radial-gradient(circle at 20% 0,#fff7e6 0,#eee6d6 38%,#ded1b9 100%)}
.app{display:grid;grid-template-columns:minmax(420px,620px) minmax(320px,1fr);gap:18px;max-width:1220px;margin:0 auto;padding:18px}
.board-shell{padding:14px;background:#7b4a22;border:1px solid #4f2d12;border-radius:8px;box-shadow:0 16px 32px #2b170833}
.board{position:relative;display:grid;grid-template-columns:repeat(9,1fr);aspect-ratio:9/10;background:#dca85a;border:3px solid #3e250e;overflow:hidden}
.board::before{content:"楚河          漢界";position:absolute;left:0;right:0;top:50%;height:9.8%;transform:translateY(-50%);display:flex;align-items:center;justify-content:center;letter-spacing:.75em;font-size:clamp(15px,2.2vw,28px);font-weight:700;color:#6d421c99;background:linear-gradient(90deg,#e7bd76,#f0cb8b 50%,#e7bd76);border-top:2px solid #80531f;border-bottom:2px solid #80531f;pointer-events:none;z-index:0}
.sq{position:relative;z-index:1;border:1px solid #7a4d1e;background:transparent;display:grid;place-items:center;min-width:0;cursor:pointer;padding:0}
.sq::after{content:attr(data-sq);position:absolute;left:4px;top:3px;font-size:10px;color:#5d371599}
.sq:hover{background:#fff3c24d}.selected{background:#ffe08299}.target{background:#b7e4c799}.capture{background:#ef9a9a99}
.piece{width:76%;height:76%;border-radius:999px;display:grid;place-items:center;font-family:"SimSun","Songti SC",serif;font-weight:900;font-size:clamp(20px,4.2vw,34px);background:radial-gradient(circle at 35% 28%,#fff9e8,#f5dfb7 65%,#d8aa64);border:2px solid #40250e;box-shadow:inset 0 0 0 2px #fff4d2,0 3px 7px #1f120840}
.piece.red{color:#b31313}.piece.black{color:#14100c}.piece::before{content:"";position:absolute}
.panel{background:#fffaf0;border:1px solid #cfbd9c;border-radius:8px;padding:14px;box-shadow:0 8px 20px #2b170812}
.title{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:10px}.title h1{font-size:22px;margin:0}.title span{font-size:13px;color:#72573a}
.stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.stat{background:#f7eddc;border:1px solid #dfcaa6;border-radius:6px;padding:8px;font-size:13px}.stat b{display:block;color:#74512b;margin-bottom:2px}
.row{display:flex;gap:8px;flex-wrap:wrap;margin:9px 0}button,input,textarea{font:inherit}button{border:1px solid #9f7a4b;background:#fff7e8;border-radius:6px;padding:7px 11px;cursor:pointer;color:#35210f}button:hover{background:#f3dfbd}
input,textarea{border:1px solid #b89260;border-radius:6px;padding:8px;width:100%;box-sizing:border-box;background:#fffdf8}textarea{height:76px;resize:vertical}
.move-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px}.msg{min-height:22px;color:#23506b}.fen{font-family:Consolas,monospace;font-size:12px;word-break:break-all;background:#fbf2e0;border:1px solid #e0c99d;border-radius:6px;padding:8px}
.moves{max-height:340px;overflow:auto;font-family:Consolas,"Microsoft YaHei",sans-serif;font-size:13px;line-height:1.55;background:#fbf2e0;border:1px solid #e0c99d;border-radius:6px;padding:8px}.moves div{display:inline-block;width:76px}.moves .cap{color:#b31313;font-weight:700}
@media(max-width:900px){.app{grid-template-columns:1fr;padding:10px}.board-shell,.panel{padding:10px}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
</head>
<body>
<div id="app" class="app">
  <div class="board-shell">
    <div class="board" aria-label="象棋棋盘">
      <button v-for="cell in flatBoard" :key="cell.sq" class="sq" :data-sq="cell.sq" :class="squareClass(cell)" @click="clickSquare(cell.sq)">
        <div v-if="cell.piece" class="piece" :class="cell.color" :title="cell.sq + ' ' + cell.kind">{{ cell.piece }}</div>
      </button>
    </div>
  </div>
  <div class="panel">
    <div class="title">
      <h1>规则棋盘</h1>
      <span>点击棋子和落点即可走子</span>
    </div>
    <div class="stats">
      <div v-for="item in stats" class="stat"><b>{{ item.key }}</b>{{ item.value }}</div>
    </div>
    <p class="fen">{{ state?.fen }}</p>
    <p class="msg">{{ state?.message || '' }}</p>
    <div class="row">
      <button @click="refresh">刷新</button>
      <button @click="undo">撤销</button>
      <button @click="resetBoard">初始局面</button>
      <button @click="copyFen">复制 FEN</button>
    </div>
    <div class="move-row">
      <input v-model="moveInput" placeholder="输入着法，例如 h2e2" @keydown.enter="playInput">
      <button @click="playInput">走子</button>
    </div>
    <textarea v-model="fenInput" placeholder="粘贴 FEN，或输入 startpos"></textarea>
    <div class="row">
      <button @click="loadFen">载入局面</button>
    </div>
    <h3>合法着法</h3>
    <div class="moves">
      <div v-for="move in state?.legalMoves || []" :class="{cap:move.capture}">{{ move.capture ? '吃 ' : '走 ' }}{{ move.uci }}</div>
    </div>
  </div>
</div>
<script>
const { createApp } = Vue;
createApp({
  data(){return{state:null,moveInput:'',fenInput:''}},
  computed:{
    flatBoard(){return (this.state?.board||[]).flat()},
    targetMap(){return new Map((this.state?.selectedMoves||[]).map(m=>[m.to,m]))},
    stats(){const s=this.state||{};return [
      {key:'行棋方',value:s.side||'-'},
      {key:'半回合',value:s.halfmove??'-'},
      {key:'基础合法着法',value:s.legalCount??'-'},
      {key:'规则合法着法',value:s.ruleLegalCount??'-'},
      {key:'红方被将',value:this.yesNo(s.redCheck)},
      {key:'黑方被将',value:this.yesNo(s.blackCheck)},
      {key:'规则结果',value:s.ruleOutcome||'未结束'},
      {key:'选中',value:s.selected||'无'}
    ]}
  },
  methods:{
    yesNo(v){return v===undefined?'-':(v?'是':'否')},
    async api(path,opts={}){const r=await fetch(path,opts);const j=await r.json();if(!r.ok)throw new Error(j.error||r.statusText);return j},
    setState(s){this.state=s;this.fenInput=s.fen},
    async refresh(){this.setState(await this.api('/api/state'))},
    async clickSquare(sq){this.setState(await this.api('/api/click',{method:'POST',body:sq}))},
    async playInput(){const v=this.moveInput.trim();if(!v)return;this.setState(await this.api('/api/move',{method:'POST',body:v}));this.moveInput=''},
    async undo(){this.setState(await this.api('/api/undo',{method:'POST'}))},
    async resetBoard(){this.setState(await this.api('/api/reset',{method:'POST'}))},
    async loadFen(){this.setState(await this.api('/api/load',{method:'POST',body:this.fenInput.trim()||'startpos'}))},
    async copyFen(){await navigator.clipboard.writeText(this.state.fen);this.state.message='FEN 已复制'},
    squareClass(cell){const target=this.targetMap.get(cell.sq);return{selected:cell.sq===this.state?.selected,target:target&&!target.capture,capture:target&&target.capture}}
  },
  mounted(){this.refresh().catch(err=>alert(err.message))}
}).mount('#app');
</script>
</body>
</html>
"#;

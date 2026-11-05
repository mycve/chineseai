// 棋子组索引复用 ChessDB；仅初始化和 FFI 封装为本项目新增。
#include "xiangqi.h"
#include "piece_set.h"
#include <set>
#include <algorithm>
#include <errno.h>

using namespace std;

const U8* Piece_Set_ID[16] = {
	NULL, NULL, Rook_Set_ID[White], Knight_Set_ID[White], Cannon_Set_ID[White], NULL, NULL, Pawn_Set_ID[White],
	NULL, NULL, Rook_Set_ID[Black], Knight_Set_ID[Black], Cannon_Set_ID[Black], NULL, NULL, Pawn_Set_ID[Black],
};

S8 Piece_Sq_Index[1440];
Group_Info Group_Table[40];

void piece_index_init()
{
	int ix, data;
	memset(Piece_Sq_Index, 0, sizeof(Piece_Sq_Index));
	for (int piece = 1; piece < 16; ++piece)
	{
		if (piece == 8)
			continue;
		for (int sq = 0; sq < 90; ++sq)
		{
			ix = (sq << 4) + piece;
			switch (piece)
			{
			case WhitePawn:
				data = White_Pawn_Pos[sq];
				break;
			case BlackPawn:
				data = White_Pawn_Pos[sq_rank_mirror(sq)];
				break;
			case WhiteKing:
				data = White_King_Pos[sq];
				break;
			case BlackKing:
				data = White_King_Pos[sq_rank_mirror(sq)];
				break;
			case WhiteAdvisor:
				data = White_Advisor_Pos[sq];
				break;
			case BlackAdvisor:
				data = White_Advisor_Pos[sq_rank_mirror(sq)];
				break;
			case WhiteBishop:
				data = White_Bishop_Pos[sq];
				break;
			case BlackBishop:
				data = White_Bishop_Pos[sq_rank_mirror(sq)];
				break;
			default:
				data = sq;
			}
			Piece_Sq_Index[ix] = data;
		}
	}
}


static void calc_egtb_index(Position& pos, char* names, S8& mirror, U64& offset)
{
    S8 piece;
    S8 piece_list[16][8] = {0};

    int scores[2] = {0};
    for (int i = 0; i < 2; ++i)
    {
        for (int j = King; j <= Pawn; ++j)
        {
            S8 sq;
            int cnt = 0;
            piece = piece_make(i, j);
            BITBOARD pieces_bb = pos.pieces[piece];
            while (pieces_bb)
            {
                sq = pieces_bb.pop_1st_sq();
                piece_list[piece][cnt++] = sq;
                scores[i] += Piece_Order_Value[piece];
            }
        }
    }
    mirror = (scores[1] > scores[0] || (scores[1] == scores[0] && pos.turn == Black)) ? 1 : 0;

    S8 set_id[10] = {0};
    S8 square_list[10][16] = {0};
    S8 ix, id;
    size_t nam_ix = 0;

    if (mirror)
    {
        for (int i = 0; i < 2; ++i)
        {
            ix = 0;
            S8 color = color_opp(i);
            id = WSet_Defend + 5*i;
            set_id[id] = defend_set(pos.number[piece_make(color, Advisor)], pos.number[piece_make(color, Bishop)], i);
            piece = piece_make(color, King);
            square_list[id][ix++] = sq_rank_mirror(piece_list[piece][0]);
            names[nam_ix++] = piece_to_char(King);

            piece = piece_make(color, Advisor);
            for (int j = 0; j < pos.number[piece]; ++j)
            {
                square_list[id][ix++] = sq_rank_mirror(piece_list[piece][j]);
                names[nam_ix++] = piece_to_char(Advisor);
            }
            piece = piece_make(color, Bishop);
            for (int j = 0; j < pos.number[piece]; ++j)
            {
                square_list[id][ix++] = sq_rank_mirror(piece_list[piece][j]);
                names[nam_ix++] = piece_to_char(Bishop);
            }
            piece = piece_make(color, Rook);
            if (pos.number[piece])
            {
                ix = 0;
                id = WSet_Rook + 5*i;
                set_id[id] = Rook_Set_ID[i][pos.number[piece]];
                for (int j = 0; j < pos.number[piece]; ++j)
                {
                    square_list[id][ix++] = sq_rank_mirror(piece_list[piece][j]);
                    names[nam_ix++] = piece_to_char(Rook);
                }
            }
            piece++;
            if (pos.number[piece])
            {
                ix = 0;
                id = WSet_Knight + 5*i;
                set_id[id] = Knight_Set_ID[i][pos.number[piece]];
                for (int j = 0; j < pos.number[piece]; ++j)
                {
                    square_list[id][ix++] = sq_rank_mirror(piece_list[piece][j]);
                    names[nam_ix++] = piece_to_char(Knight);
                }
            }
            piece++;
            if (pos.number[piece])
            {
                ix = 0;
                id = WSet_Cannon + 5*i;
                set_id[id] = Cannon_Set_ID[i][pos.number[piece]];
                for (int j = 0; j < pos.number[piece]; ++j)
                {
                    square_list[id][ix++] = sq_rank_mirror(piece_list[piece][j]);
                    names[nam_ix++] = piece_to_char(Cannon);
                }
            }
            piece += 3;
            if (pos.number[piece])
            {
                ix = 0;
                id = WSet_Pawn + 5*i;
                set_id[id] = Pawn_Set_ID[i][pos.number[piece]];
                for (int j = 0; j < pos.number[piece]; ++j)
                {
                    square_list[id][ix++] = sq_rank_mirror(piece_list[piece][j]);
                    names[nam_ix++] = piece_to_char(Pawn);
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < 2; ++i)
        {
            ix = 0;
            id = WSet_Defend + 5*i;
            set_id[id] = defend_set(pos.number[piece_make(i, Advisor)], pos.number[piece_make(i, Bishop)], i);
            piece = piece_make(i, King);
            square_list[id][ix++] = piece_list[piece][0];
            names[nam_ix++] = piece_to_char(King);

            piece = piece_make(i, Advisor);
            for (int j = 0; j < pos.number[piece]; ++j)
            {
                square_list[id][ix++] = piece_list[piece][j];
                names[nam_ix++] = piece_to_char(Advisor);
            }
            piece = piece_make(i, Bishop);
            for (int j = 0; j < pos.number[piece]; ++j)
            {
                square_list[id][ix++] = piece_list[piece][j];
                names[nam_ix++] = piece_to_char(Bishop);
            }
            piece = piece_make(i, Rook);
            if (pos.number[piece])
            {
                ix = 0;
                id = WSet_Rook + 5*i;
                set_id[id] = piece_set(piece, pos.number[piece]);
                for (int j = 0; j < pos.number[piece]; ++j)
                {
                    square_list[id][ix++] = piece_list[piece][j];
                    names[nam_ix++] = piece_to_char(Rook);
                }
            }
            piece++;
            if (pos.number[piece])
            {
                ix = 0;
                id = WSet_Knight + 5*i;
                set_id[id] = piece_set(piece, pos.number[piece]);
                for (int j = 0; j < pos.number[piece]; ++j)
                {
                    square_list[id][ix++] = piece_list[piece][j];
                    names[nam_ix++] = piece_to_char(Knight);
                }
            }
            piece++;
            if (pos.number[piece])
            {
                ix = 0;
                id = WSet_Cannon + 5*i;
                set_id[id] = piece_set(piece, pos.number[piece]);
                for (int j = 0; j < pos.number[piece]; ++j)
                {
                    square_list[id][ix++] = piece_list[piece][j];
                    names[nam_ix++] = piece_to_char(Cannon);
                }
            }
            piece += 3;
            if (pos.number[piece])
            {
                ix = 0;
                id = WSet_Pawn + 5*i;
                set_id[id] = piece_set(piece, pos.number[piece]);
                for (int j = 0; j < pos.number[piece]; ++j)
                {
                    square_list[id][ix++] = piece_list[piece][j];
                    names[nam_ix++] = piece_to_char(Pawn);
                }
            }
        }
    }
    names[nam_ix] = 0;

    int compress_id = 0;
    const Group_Info* info;
    float best_ratio = 100.0f;
    for (int i = 0; i < 10; ++i)
    {
        if (set_id[i] == 0) continue;
        info = get_set_info(set_id[i]);
        const float ratio = (float)info->compress_size / info->table_size;
        if (ratio < best_ratio) { best_ratio = ratio; compress_id = i; }
    }

    offset = 0ULL;
    info = get_set_info(set_id[compress_id]);
    U64 weight = 1;
    const U32 compress_ix = info->get_list_pos(square_list[compress_id]);

    if ((compress_ix & 0xffffU) < info->compress_size)
    {
        for (int i = 0; i < 10; ++i)
        {
            if (set_id[i] == 0) continue;
            info = get_set_info(set_id[i]);
            if (i == compress_id) {
                offset += (compress_ix & 0xffffU) * weight;
                weight *= info->compress_size;
            } else {
                offset += (info->get_list_pos(square_list[i]) & 0xffffU) * weight;
                weight *= info->table_size;
            }
        }
    }
    else
    {
        for (int i = 0; i < 10; ++i)
        {
            if (set_id[i] == 0) continue;
            info = get_set_info(set_id[i]);
            if (i == compress_id) {
                offset += ((compress_ix >> 16) & 0xffffU) * weight;
                weight *= info->compress_size;
            } else {
                const U32 v = info->get_list_pos(square_list[i]);
                offset += ((v >> 16) & 0xffffU) * weight;
                weight *= info->table_size;
            }
        }
    }
}


#include <mutex>
#include <stdexcept>
#include <cstdint>
static std::once_flag init_once;
extern "C" int chineseai_tb_init(const unsigned char* data, size_t size) {
    try {
        std::call_once(init_once, [=] {
            size_t offset = 0;
            for (auto& group : Group_Table) {
                if (size - offset < 28) throw std::runtime_error("GROUP header truncated");
                std::memcpy(&group, data + offset, 28); offset += 28;
                if (group.piece_cnt < 0 || group.piece_cnt > 6) throw std::runtime_error("GROUP invalid");
                if (group.piece_cnt) {
                    const size_t bytes = size_t(group.index_size) * 4;
                    if (bytes > size - offset) throw std::runtime_error("GROUP index truncated");
                    group.index_tb = new uint32[group.index_size];
                    std::memcpy(group.index_tb, data + offset, bytes); offset += bytes;
                }
            }
            piece_index_init(); attack_init();
        });
        return 1;
    } catch (...) { return 0; }
}
extern "C" int chineseai_tb_index(const char* fen, char* names, uint64_t* offset, int* side) {
    Position pos;
    if (!pos.from_fen(fen) || pos.number[WhiteKing] != 1 || pos.number[BlackKing] != 1
        || !pos.is_legal() || pos.number[WhitePawn] >= 4 || pos.number[BlackPawn] >= 4) return 0;
    for (int color = 0; color < 2; ++color) {
        for (int kind = Rook; kind <= Bishop; ++kind)
            if (pos.number[piece_make(color, kind)] > 2) return 0;
    }
    for (int square = 0; square < 90; ++square)
        if (pos.square[square] && piece_index(pos.square[square], square) < 0) return 0;
    S8 mirror; U64 index;
    calc_egtb_index(pos, names, mirror, index);
    *offset = index; *side = mirror ? color_opp(pos.turn) : pos.turn;
    return 1;
}

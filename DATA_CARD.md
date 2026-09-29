# Data card: XiangqiBench positions

## Summary

`src/xiangqibench/data/cases.jsonl` holds 119 composed xiangqi endgames. In every one, Red is to
move and can force checkmate. Load them with `xiangqibench.load_cases()` or list them with
`xiangqibench cases`.

| | |
|---|---|
| Positions | 119 (116 *Shi Qing Ya Qu*, 3 *Jianghu*) |
| Side to move | Red in all positions (the agent plays Red) |
| Stored mate distance | 3–11 plies, median 9 |
| Categories | 106 `threatmate_race`, 13 `forced_mate` |
| Admission | 117 confirmed by Pikafish (`verified_by: engine`), 2 by a checks-only mate search (`solver_proof`) |
| sha256 | `3cb4e29148f82cb6d9a881e28336676549225cfe86dd8fb947ccf580f1633d42` |

## Provenance

- ***Shi Qing Ya Qu*** (适情雅趣) is a Ming-dynasty treatise of composed endgames, printed in
  1570. Digitized entries were taken from
  [xqipu.com](https://www.xqipu.com/canjugupu/1547). Case ids are
  `xq_shi_qing_ya_qu_<n>`, and `raw_index` is the entry's position in that digitization.
- ***Jianghu*** (江湖) is a classical collection of street endgames. Case ids are
  `xq_jianghu_endgames_<n>`.

Both sources are several centuries old and are in the public domain. The historical solution
text is not included, is never shown to agents, and is not used for scoring. They may still
appear in pre-training corpora, so first-move agreement can partly reflect recall.

## Admission

Source solutions were not trusted. Each position was screened with a single-threaded,
time-limited Pikafish search and admitted when the engine reported a forced mate for Red. The
engine's mate distance and preferred first move are stored as `engine_mate_plies` and
`first_winning_move`.

Pikafish does not confirm a mate in two compositions, `xq_jianghu_endgames_084` and
`xq_jianghu_endgames_327`. They were admitted by a checks-only mate search and carry
`verified_by: solver_proof`. All 119 positions were also checked by hand.
Admission relies on bounded search, so the positions are *search-supported* rather than formally
proved.

**Categories.** The same search was run with Black to move. If Black then also has a forced mate
(`defender_threat_in_plies`), the position is a `threatmate_race`: a slow Red move can let Black
mate first. Otherwise it is a `forced_mate`.

## Fields

| Field | Meaning |
|---|---|
| `id` | Stable case id |
| `source`, `raw_index`, `name` | Source collection, index in the digitization, original title |
| `fen` | Start position (FEN, Red to move) |
| `challenger` | Side played by the agent (`red`) |
| `category` | `threatmate_race` or `forced_mate` |
| `win_in_plies` | Stored mate distance (engine or solver) |
| `engine_mate_plies` | Pikafish mate distance, or `null` if the engine does not confirm a mate |
| `defender_threat_in_plies` | Black's mate distance if Black were to move, or `null` |
| `first_winning_move` | Reference root move from the engine (used only for first-move analysis) |
| `legal_move_count` | Number of legal Red moves at the root |
| `verified_by` | `engine` or `solver_proof` |
| `difficulty_score`, `tier`, `piece_theme`, `tags` | Descriptive metadata; not used for scoring |

The stored distances are admission metadata. They do not limit the game: every trial may run up
to 40 plies.

## Splits

| Split | Cases | Use |
|---|---|---|
| `main` | 119 | Leaderboard (paper settings `sighted` and `restricted`) |
| `ablation-gemini-3.1-pro` | 20 | Observation ablation for Gemini 3.1 Pro |
| `ablation-gpt-5.5` | 21 | Observation ablation for GPT-5.5 |

The ablation subsets are **model-specific**. Each contains positions that the model won at
least once in the paper's Sighted trials, so the ablation measures how much of a demonstrated
ability survives each observation change.

- For GPT-5.5 the subset contains every such position.
- For Gemini 3.1 Pro it is a stratified sample by mate distance, with a quota
  {5: 3, 7: 6, 9: 5, 11: 6} and seed 20260926.

Ablation numbers are therefore not comparable across models or with the `main` split.

## Engine coverage

Pikafish refuses to search two positions reachable in the benchmark: the start positions of
`xq_jianghu_endgames_084` and `xq_jianghu_endgames_327`, and the positions after their first
move. For such positions, the defender falls back to a depth-5 rule search. Each defender move
records which backend chose it, so the share of rule-chosen moves can be computed from any set
of records.

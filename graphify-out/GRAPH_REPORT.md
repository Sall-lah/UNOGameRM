# Graph Report - .  (2026-10-21)

## Corpus Check
- Corpus is ~23,329 words - fits in a single context window. You may not need a graph.

## Summary
- 87 nodes · 178 edges · 8 communities (7 shown, 1 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 28 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_In-Game TUI Rendering|In-Game TUI Rendering]]
- [[_COMMUNITY_Application Lifecycle & Build|Application Lifecycle & Build]]
- [[_COMMUNITY_Card & Deck Mechanics|Card & Deck Mechanics]]
- [[_COMMUNITY_CLI Input Handling & Validation|CLI Input Handling & Validation]]
- [[_COMMUNITY_Player & AI Decision Logic|Player & AI Decision Logic]]
- [[_COMMUNITY_Game Setup & Instructions|Game Setup & Instructions]]
- [[_COMMUNITY_Victory Screen & Progress Animations|Victory Screen & Progress Animations]]
- [[_COMMUNITY_Unix Install Script|Unix Install Script]]

## God Nodes (most connected - your core abstractions)
1. `Deck` - 13 edges
2. `Game` - 12 edges
3. `Card` - 10 edges
4. `inputHandler()` - 10 edges
5. `InputError` - 9 edges
6. `assignInputToVar()` - 9 edges
7. `AIPlayer` - 9 edges
8. `Player` - 8 edges
9. `print_heading()` - 7 edges
10. `display_players_and_top_card()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Install Script` --semantically_similar_to--> `PyInstaller CI Workflow`  [INFERRED] [semantically similar]
  install.sh → .github/workflows/pyinstaller.yml
- `UNO Call Declaration` --handled_by--> `get_card_choice()`  [INFERRED]
  README Images/declared_uno_and reversed_turn_order.png → cli.py
- `Reboot Progress Bar UI` --conceptually_related_to--> `progress_bar()`  [INFERRED]
  README Images/lesgo_i_win.png → cli.py
- `Boot Progress Indicator` --references--> `progress_bar()`  [INFERRED]
  README Images/setup_screen.png → cli.py
- `UNO.PY Header Banner` --rendered_by--> `print_heading()`  [INFERRED]
  README Images/declared_uno_and reversed_turn_order.png → cli.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Turn Execution and Rendering Loop** — uno_game_play_turn, player_aiplayer_think, cli_display_players_and_top_card, cli_display_hand [INFERRED 0.85]
- **UNO Game Core Domain Model** — card_card, deck_deck, player_player, uno_game [INFERRED 0.95]
- **CLI Input Sanitization and Choice Validation** — cli_tools_inputhandler, cli_tools_assigninputtovar, cli_get_card_choice, cli_tools_inputerror [INFERRED 0.85]
- **Game Over and Victory UI Sequence** — readme_images_lesgo_i_win_victory_screen, readme_images_lesgo_i_win_ascii_victory_art, readme_images_lesgo_i_win_reboot_progress_bar, readme_images_lesgo_i_win_exit_prompt [EXTRACTED 1.00]
- **Game Setup and Initialization Flow** — readme_images_setup_screen_ui, readme_images_setup_screen_boot_progress, readme_images_setup_screen_console, uno_game_setup [INFERRED 0.85]
- **TUI In-Game Screen Layout** — readme_images_variety_of_cards_in_hand, readme_images_variety_of_cards_in_hand_cards_display, readme_images_variety_of_cards_in_hand_players_and_discard, readme_images_variety_of_cards_in_hand_console_status [EXTRACTED 1.00]

## Communities (8 total, 1 thin omitted)

### Community 0 - "In-Game TUI Rendering"
Cohesion: 0.17
Nodes (15): display_hand(), display_players_and_top_card(), trueColour(), Game Event Console Log, Active Discard Top Card Display, Player Status and Card Counts Panel, Player Hand Card Box UI, Reverse Card Turn Direction Inversion (+7 more)

### Community 1 - "Application Lifecycle & Build"
Cohesion: 0.18
Nodes (11): clear(), Clears the console or terminal. Should work for all major OS types.      Can a, Attempts to automatically toggle fullscreen and move the mouse cursor out of sig, toggleFullscreen(), Conda Environment, Install Script, main, Pip Requirements (+3 more)

### Community 2 - "Card & Deck Mechanics"
Cohesion: 0.26
Nodes (3): Card, Deck.build_deck, Deck

### Community 3 - "CLI Input Handling & Validation"
Cohesion: 0.36
Nodes (10): get_card_choice(), get_colour_choice(), get_draw_4_choice(), assignInputToVar(), InputError, inputHandler(), Function to be used in conjunction with *inputHandler*.      Helps in doing wh, Custom error for use in *inputHandler*, as well as in functions used as paramete (+2 more)

### Community 4 - "Player & AI Decision Logic"
Cohesion: 0.24
Nodes (4): Card.is_playable_on, spinner(), AIPlayer, Player

### Community 5 - "Game Setup & Instructions"
Cohesion: 0.29
Nodes (9): print_heading(), print_instructions(), Deck.draw, Deck.reshuffle_discard_pile, Deck.shuffle, UNO.PY Header Banner, Player Setup Console, Gameplay Instructions Display (+1 more)

### Community 6 - "Victory Screen & Progress Animations"
Cohesion: 0.32
Nodes (7): print_victory_msg(), progress_bar(), ASCII Victory Art Display, Game Exit Input Prompt, Reboot Progress Bar UI, CLI Victory Screen Screenshot, Boot Progress Indicator

## Knowledge Gaps
- **6 isolated node(s):** `install.sh script`, `Conda Environment`, `VSVersionInfo`, `Turn Order Arrows and Active Turn Indicator`, `Game Exit Input Prompt` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Game` connect `Application Lifecycle & Build` to `Card & Deck Mechanics`, `CLI Input Handling & Validation`, `Player & AI Decision Logic`, `Game Setup & Instructions`, `Victory Screen & Progress Animations`?**
  _High betweenness centrality (0.163) - this node is a cross-community bridge._
- **Why does `Deck` connect `Card & Deck Mechanics` to `Application Lifecycle & Build`, `CLI Input Handling & Validation`, `Player & AI Decision Logic`, `Game Setup & Instructions`?**
  _High betweenness centrality (0.132) - this node is a cross-community bridge._
- **Why does `display_hand()` connect `In-Game TUI Rendering` to `Card & Deck Mechanics`, `CLI Input Handling & Validation`, `Player & AI Decision Logic`, `Game Setup & Instructions`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `Deck` (e.g. with `Card` and `Player`) actually correct?**
  _`Deck` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `Game` (e.g. with `InputError` and `Deck`) actually correct?**
  _`Game` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Card` (e.g. with `display_hand()` and `display_players_and_top_card()`) actually correct?**
  _`Card` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Custom error for use in *inputHandler*, as well as in functions used as paramete`, `Clears the console or terminal. Should work for all major OS types.      Can a`, `Attempts to automatically toggle fullscreen and move the mouse cursor out of sig` to the rest of the system?**
  _11 weakly-connected nodes found - possible documentation gaps or missing edges._
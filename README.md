# blackjack-sim
Simulates blackjack card counting EV using Hi-Lo.

To run: `python3 sim.py --parallel n` to run n shoes in parallel (utilizes the maximum number of threads possible)

Options:
- `--h17`  to simulate dealer hitting soft 17
- `--decks n` to set n decks (default 6)
- `--pen n` to set penetration to n (default 0.75)
- `--min_bet n` to set minimum bet to n (TODO: custom bet spreading. Default is 1 to 9)
- `--wong_out n` to set TC threshold at which the shoe ends (default -2)
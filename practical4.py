tree = {
    "A": ["B", "C"], "B": ["D", "E"], "C": ["F", "G"],
    "D": ["H", "I"], "E": ["J", "K"],
    "F": ["L", "M"], "G": ["N", "O"]
}
payoffs = {"H": 4, "I": 9, "J": 7, "K": 2,
           "L": 3, "M": 8, "N": 6, "O": 1}


def is_terminal(state):
    return state in payoffs


def successors(state, reverse=False):
    children = tree[state]
    return list(reversed(children)) if reverse else children[:]


def utility(state):
    return payoffs[state]


def minimax(state, maximizing, counter, reverse=False):
    counter["calls"] += 1
    if is_terminal(state):
        return utility(state)
    values = []
    for child in successors(state, reverse):
        values.append(minimax(child, not maximizing, counter, reverse))
    return max(values) if maximizing else min(values)


def alphabeta(state, alpha, beta, maximizing, counter, reverse=False):
    counter["calls"] += 1
    counter["visited"].append(state)
    if is_terminal(state):
        return utility(state)
    # Шектерді түйінге кірген сәтте сақтаймыз.
    row = {"node": state, "alpha": alpha, "beta": beta,
           "children": [], "pruned": []}
    counter["trace"].append(row)
    value = float("-inf") if maximizing else float("inf")
    children = successors(state, reverse)
    for i, child in enumerate(children):
        row["children"].append(child)
        result = alphabeta(child, alpha, beta, not maximizing,
                           counter, reverse)
        if maximizing:
            value = max(value, result)
            alpha = max(alpha, value)
        else:
            value = min(value, result)
            beta = min(beta, value)
        if alpha >= beta:
            row["pruned"] = children[i + 1:]
            counter["pruned"].extend(children[i + 1:])
            break
    row["value"] = value
    return value


def run_search(reverse=False):
    mm = {"calls": 0}
    ab = {"calls": 0, "visited": [], "pruned": [], "trace": []}
    m = minimax("A", True, mm, reverse)
    a = alphabeta("A", float("-inf"), float("inf"), True, ab, reverse)
    assert m == a == 7
    print("Order:", "right-to-left" if reverse else "left-to-right")
    print("Minimax: value =", m, "; calls =", mm["calls"])
    print("Alpha-Beta: value =", a, "; calls =", ab["calls"])
    print("Visited:", ", ".join(ab["visited"]))
    print("Pruned:", ", ".join(ab["pruned"]) or "none")
    print()
    return ab


features = {"X": (8, 7, 2), "Y": (5, 5, 8)}


def evaluate(state, weights):
    margin, reliability, capacity = features[state]
    return (weights[0] * margin + weights[1] * reliability
            + weights[2] * capacity)


def run_evaluation():
    for weights in [(0.5, 0.3, 0.2), (0.3125, 0.1875, 0.5)]:
        scores = {s: evaluate(s, weights) for s in features}
        assert abs(sum(weights) - 1) < 1e-9
        print("Weights:", weights)
        print("X =", round(scores["X"], 4),
              "; Y =", round(scores["Y"], 4))
        print("MAX chooses:", max(scores, key=scores.get))
    print()


delivery = {"A": [8, 6, 4], "B": [3, 7, 5], "C": [2, 9, 1]}
probabilities = {"A": [0.2, 0.3, 0.5],
                 "B": [0.2, 0.5, 0.3], "C": [0.7, 0.2, 0.1]}


def chance_value(strategy):
    total = 0
    for p, value in zip(probabilities[strategy], delivery[strategy]):
        total += p * value
    return total


def run_expectimax():
    expected = {}
    worst = {}
    for s in delivery:
        assert abs(sum(probabilities[s]) - 1) < 1e-9
        expected[s] = chance_value(s)
        worst[s] = min(delivery[s])
        print(s, "expected =", round(expected[s], 2),
              "; minimum =", worst[s])
    print("Expectimax chooses:", max(expected, key=expected.get))
    print("Minimax chooses:", max(worst, key=worst.get))


def main():
    run_search(False)
    run_search(True)
    run_evaluation()
    run_expectimax()


if __name__ == "__main__":
    main()

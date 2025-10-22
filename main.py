from classes import Grid, ArcTask
from utils import load_all, load_arc1, load_arc2

def print_prob(prob: ArcTask, test_only: bool = True) -> None:
    train, test = ("Train", prob.train_pairs), ("Test", prob.test_pairs)
    sets = (test,) if test_only else (test, train)
    for name, pairs in sets:
        print(f"\n{name} pairs:")
        for pair in pairs:
            inp = Grid.from_arr(pair.input)
            out = Grid.from_arr(pair.output)
            print(f"\nInput:\n{inp}\nOutput:\n{out}")

def main():
    all_arc = load_arc1()
    ez_size = [prob.size_heuristic() for prob in all_arc.training + all_arc.evaluation]

    ez = ez_size.count(True)

    print(f"{ez}/{len(ez_size)} problems have easy size")

    while True:
        inp = "Enter problem name / hash > "
        problem_hash = input(inp).strip()

        try:
            prob = all_arc.get_problem(problem_hash)
        except ValueError:
            print(f"Problem '{problem_hash}' not found")
            continue
        
        print(f"Size: {"easy" if prob.size_heuristic() else "hard"}")
        print_prob(prob)

if __name__ == "__main__":
    main()
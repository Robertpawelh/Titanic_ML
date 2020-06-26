import pandas as pd

def prepare_solution(ids, prediction, name='solution'):
    solution = pd.DataFrame({
        'PassengerId': ids,
        'Survived': prediction
    })

    solution.to_csv(f'{name}.csv', index=False)
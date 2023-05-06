import collections
from nltk.metrics.scores import accuracy, precision, recall, f_measure        

def report_results(classifer_type, y_test_lst, results):
    print(classifer_type, "Accuracy on test set", accuracy(y_test_lst, results))

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (ref, pred) in enumerate(zip(y_test_lst, results)):
        refsets[ref].add(i)
        testsets[pred].add(i)

    print(classifer_type, "precision on True label", precision(refsets[True], testsets[True]))
    print(classifer_type, "recall on True label", recall(refsets[True], testsets[True]))
    print(classifer_type, "f measure on True label", f_measure(refsets[True], testsets[True]))

    print(classifer_type, "precision on False label", precision(refsets[False], testsets[False]))
    print(classifer_type, "recall on False label", recall(refsets[False], testsets[False]))
    print(classifer_type, "f measure on False label", f_measure(refsets[False], testsets[False]))

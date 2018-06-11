import pandas as pd
import numerox as nx


def cv_warning(data, tournament='bernie', nsamples=100):
    "Hold out a sample of eras not rows when doing cross validation."

    data = data['train']
    model = nx.logistic()
    results_cve = pd.DataFrame()
    results_cv = pd.DataFrame()

    for i in range(nsamples):

        # cv across eras
        cve = nx.CVSplitter(data, seed=i)
        prediction = nx.run(model, cve, tournament, verbosity=0)
        df = prediction.performance(data, tournament)
        results_cve = results_cve.append(df, ignore_index=True)

        # cv ignoring eras but y balanced
        cv = nx.IgnoreEraCVSplitter(data, tournament=tournament, seed=i)
        prediction = nx.run(model, cv, tournament, verbosity=0)
        df = prediction.performance(data, tournament)
        results_cv = results_cv.append(df, ignore_index=True)

        # display results
        rcve = results_cve.mean(axis=0)
        rcv = results_cv.mean(axis=0)
        rcve.name = 'cve'
        rcv.name = 'cv'
        r = pd.concat([rcve, rcv], axis=1)
        print("\n{} runs".format(i+1))
        print(r)

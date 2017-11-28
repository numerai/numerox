#!/usr/bin/env python

import pandas as pd
import numerox as nx


def cv_warning(data, nsamples=100):

    model = nx.model.logistic()

    for i in range(nsamples):

        report = nx.Report()

        # cv across eras
        cve = nx.CVSplitter(data, seed=i)
        prediction = nx.run(model, cve, verbosity=0)
        report.append_prediction(prediction, 'cve')

        # cv ignoring eras but y balanced
        cv = nx.IgnoreEraCVSplitter(data, seed=i)
        prediction = nx.run(model, cv, verbosity=0)
        report.append_prediction(prediction, 'cv')

        # save performance results
        df = report.performance_df(data)
        cols = df.columns.tolist()
        cols[-1] = 'cv_type'
        df.columns = cols
        if i == 0:
            results = df
        else:
            results = results.append(df, ignore_index=True)

        # display results
        rcve = results[results.cv_type == 'cve'].mean(axis=0)
        rcv = results[results.cv_type == 'cv'].mean(axis=0)
        rcve.name = 'cve'
        rcv.name = 'cv'
        r = pd.concat([rcve, rcv], axis=1)
        print("\n{} runs".format(i+1))
        print(r)


if __name__ == '__main__':
    data = nx.numerai.download_data_object()
    data = data['train']
    cv_warning(data)

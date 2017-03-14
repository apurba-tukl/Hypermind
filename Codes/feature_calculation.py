import numpy as np
import pandas as pd
import os


def calcFeature(ir, fx, rg, ts):
    # returns [temperatuer_slope, temperature_std, ]
    begin_time = rg["# timestamp"].values[0] + ts
    end_time = rg["# timestamp"].values[-1] + ts
    temps = ir[(ir["# timestamp"] > begin_time) & (ir["# timestamp"] < end_time)]["nose_filtered"].values
    pupils = rg[(rg["left_pupil"] > 0)]["left_pupil"].values

    return [
            np.std(temps),
            (temps[-1] - temps[0])/(end_time - begin_time),
            np.std(pupils),
            (pupils[-1] - pupils[0])/(end_time - begin_time),
           ]


cd = os.path.dirname(os.path.abspath(__file__))
groupA = [1, 3, 6, 7, 10, 11, 12]
groupB = [2, 4, 5, 8, 9, 13, 14]
scores = pd.read_csv(cd + "/../Datafiles/scores.csv").values
# format of scores: pid, grade, group, A1_score, A1_confidence, A2...

# for ts in np.arange(0, 10, 0.5):  # time shift
for ts in np.arange(0, 10, 0.5):  # time shift
    features = []

    for p in groupA:
        ir = pd.read_csv(cd + "/../Datafiles/working/ir/" + "p" + str(p).zfill(2) + ".csv")
        answers = scores[p-1][3:19:2]
        confidences = scores[p-1][4:20:2]

        for i in range(8):  # 1: reading, 2-9: questions
            fx = pd.read_csv(cd + "/../Datafiles/working/gaze-abstime-separate-fixation/"
                             + "ap" + str(p).zfill(2) + "_" + str(i+2).zfill(2) + ".csv")
            rg = pd.read_csv(cd + "/../Datafiles/working/gaze-abstime-separate-raw/"
                             + "ap" + str(p).zfill(2) + "_" + str(i+2).zfill(2) + ".csv")
            features.append(["P"+str(p), answers[i-1], confidences[i-1]] + calcFeature(ir, fx, rg, ts))

    for p in groupB:
        ir = pd.read_csv(cd + "/../Datafiles/working/ir/" + "p" + str(p).zfill(2) + ".csv")
        answers = scores[p-1][3:19:2]
        confidences = scores[p-1][4:20:2]

        for i in range(8):  # 1: reading question, 2: reading text, 3-10: questions
            fx = pd.read_csv(cd + "/../Datafiles/working/gaze-abstime-separate-fixation/"
                             + "bp" + str(p).zfill(2) + "_" + str(i+3).zfill(2) + ".csv")
            rg = pd.read_csv(cd + "/../Datafiles/working/gaze-abstime-separate-raw/"
                             + "bp" + str(p).zfill(2) + "_" + str(i+3).zfill(2) + ".csv")
            features.append(["P"+str(p), answers[i], confidences[i]] + calcFeature(ir, fx, rg, ts))

    df = pd.DataFrame(data=np.array(features),
                      columns=["participant", "answer", "confidence", "std_temp", "slope_temp", "std_pupil", "slope_pupil"])
    df.to_csv(cd + "/../Datafiles/working/feature/shift_"+str(ts)+".csv", index=False)

import numpy as np
import os
from scipy.misc import imresize


class Evaluation:

    @staticmethod
    def evaluate_error_costs(predicted, actual, learning_len, t, save_path, test_train):

        save_path = os.path.join(save_path, test_train)
        os.makedirs(save_path, exist_ok=True)

        # Evaluate error for train data
        overall_val_costs = np.zeros(len(actual))
        for i in range(len(actual)):
            overall_val_costs[i] = np.linalg.norm(np.squeeze(predicted[i]) - np.squeeze(actual[i]))

        # Chunk reconstruction error based by individual video to process video wise
        for idx, val_costs in enumerate(overall_val_costs.reshape(-1, learning_len)):

            video_id = idx + 1
            # Save frame chunk of 't' wise reconstruction error
            file_name_prefix = 'vol_costs_{0}_video.txt'.format(video_id)
            np.savetxt(os.path.join(save_path, file_name_prefix), val_costs, delimiter=",")

            # Re-frame to actual video length
            raw_costs = imresize(np.expand_dims(val_costs, 1), (len(val_costs) + t, 1))
            raw_costs = np.squeeze(raw_costs)
            file_name_prefix = 'frame_costs_{0}_video.txt'.format(video_id)
            np.savetxt(os.path.join(save_path, file_name_prefix), raw_costs, delimiter=",")

            # Convert error score to regularised/scaled regularity score
            score_vid = raw_costs - min(raw_costs)
            score_vid = 1 - (score_vid / max(score_vid))

            file_name_prefix = 'frame_costs_scaled_{0}_video.txt'.format(video_id)
            np.savetxt(os.path.join(save_path, file_name_prefix), 1 - score_vid)

            file_name_prefix = 'regularity_scaled_{0}_video.txt'.format(video_id)
            np.savetxt(os.path.join(save_path, file_name_prefix), score_vid)

        file_name_prefix = 'overall_cost_for_all_videos.txt'
        np.savetxt(os.path.join(save_path, file_name_prefix), np.mean(overall_val_costs.reshape(-1, learning_len), axis=1), delimiter=",")


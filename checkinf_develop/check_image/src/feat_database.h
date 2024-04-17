#ifndef VISUAL_FEATURE_DATABASE_H
#define VISUAL_FEATURE_DATABASE_H

#include <Eigen/Eigen>
#include <memory>
#include <mutex>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "data_and_cal.h"

namespace Visual {

/**
 * @brief Database containing features we are currently tracking.
 */
class FeatureDatabase {

    public:
        /**
         * @brief Default constructor
         */
        FeatureDatabase() {}

        /**
         * @brief Update a feature object
         * @param id ID of the feature we will update
         * @param timestamp time that this measurement occured at
         * @param cam_id which camera this measurement was from
         * @param u raw u coordinate
         * @param v raw v coordinate
         * @param u_n undistorted/normalized u coordinate
         * @param v_n undistorted/normalized v coordinate
         *
         * This will update a given feature based on the passed ID it has.
         * It will create a new feature, if it is an ID that we have not seen before.
         */
        void update_feature(size_t id, double timestamp, float u, float v, float u_n, float v_n) {

            // Find this feature using the ID lookup
            std::unique_lock<std::mutex> lck(mtx);
            if (features_idlookup.find(id) != features_idlookup.end()) {
                // Append this new information to it!
                features_idlookup.at(id).uvs.push_back(Eigen::Vector2f(u, v));
                features_idlookup.at(id).uvs_norm.push_back(Eigen::Vector2f(u_n, v_n));
                features_idlookup.at(id).timestamps.push_back(timestamp);
                return;
            }

            // Else we have not found the feature, so lets make it be a new one!
            Visual::Feature feat;
            feat.featid = id;
            feat.uvs.push_back(Eigen::Vector2f(u, v));
            feat.uvs_norm.push_back(Eigen::Vector2f(u_n, v_n));
            feat.timestamps.push_back(timestamp);

            // Append this new feature into our database
            features_idlookup[id] = feat;
        }

        /**
         * @brief Get features that has measurements at the specified times.
         *
         * This function will return all features that have the specified time in them.
         * This would be used to get all features that occurred at a specific clone/state.
         */
        std::vector<Visual::Feature> features_contain2time(double timestamp1, double timestamp2) {

            // Our vector of old features
            std::vector<Visual::Feature> feats_has_timestamp;

            // Now lets loop through all features, and just make sure they are not
            std::unique_lock<std::mutex> lck(mtx);
            for (auto it = features_idlookup.begin(); it != features_idlookup.end();) {
                // Boolean if it has the timestamp
                // Break out if we found a single timestamp that is equal to the specified time
                bool has_timestamp1 = false, has_timestamp2 = false;
                has_timestamp1 = (std::find((*it).second.timestamps.begin(), (*it).second.timestamps.end(), timestamp1) != (*it).second.timestamps.end());
                has_timestamp2 = (std::find((*it).second.timestamps.begin(), (*it).second.timestamps.end(), timestamp2) != (*it).second.timestamps.end());
                // Remove this feature if it contains the specified timestamp
                if (has_timestamp1 && has_timestamp2) {
                    feats_has_timestamp.push_back((*it).second);
                    it++;
                } else {
                    it++;
                }
            }

            // Return the features
            return feats_has_timestamp;
        }


        /**
         * @brief Get features that has measurements at the specified time.
         *
         * This function will return all features that have the specified time in them.
         * This would be used to get all features that occurred at a specific clone/state.
         */
        std::vector<Visual::Feature> features_contain1time(double timestamp) {

            // Our vector of old features
            std::vector<Visual::Feature> feats_has_timestamp;

            // Now lets loop through all features, and just make sure they are not
            std::unique_lock<std::mutex> lck(mtx);
            for (auto it = features_idlookup.begin(); it != features_idlookup.end();) {
                // Boolean if it has the timestamp
                // Break out if we found a single timestamp that is equal to the specified time
                bool has_timestamp = false;
                has_timestamp = (std::find((*it).second.timestamps.begin(), (*it).second.timestamps.end(), timestamp) != (*it).second.timestamps.end());
                // Remove this feature if it contains the specified timestamp
                if (has_timestamp) {
                    feats_has_timestamp.push_back((*it).second);
                    it++;
                } else {
                    it++;
                }
            }

            // Return the features
            return feats_has_timestamp;
        }

    protected:

        /// Mutex lock for our map
        std::mutex mtx;

        /// Our lookup array that allow use to query based on ID
        std::unordered_map<size_t, Visual::Feature> features_idlookup;
    };

}// namespace Visual

#endif //VISUAL_FEATURE_DATABASE_H
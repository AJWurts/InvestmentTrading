from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()

## TRAIN MODEL HERE

for (market_obs_df, news_obs_df, predicions_template_df) in env.get_prediction_days():
  predictions_df = ## Algorithm Prediction Return
  env.predict(predictions_df)

env.write_submission_file()


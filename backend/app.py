
from src.api.models.prediction import lstm_prediction, gru_prediction, conv1d_prediction, ffn_prediction
from src.api.models.tools.plot import plot

# plot(lstm_prediction())
plot(lstm_prediction())
plot(ffn_prediction())
plot(gru_prediction())
plot(conv1d_prediction())

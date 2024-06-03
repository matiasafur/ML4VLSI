import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_training(plot_vars: dict):

    # Plot Results
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=plot_vars['epoch_count'], y=plot_vars['train_accuracy'], name="train_accuracy"),
                  secondary_y=False, )
    fig.add_trace(go.Scatter(x=plot_vars['epoch_count'], y=plot_vars['test_accuracy'], name="test_accuracy"),
                  secondary_y=False, )
    fig.add_trace(go.Scatter(x=plot_vars['epoch_count'], y=plot_vars['train_loss'], name="train_loss"),
                  secondary_y=True, )
    fig.add_trace(go.Scatter(x=plot_vars['epoch_count'], y=plot_vars['test_loss'], name="test_loss"),
                  secondary_y=True, )

    # Add figure title
    fig.update_layout(
        title_text="Training Progress"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Epochs")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Accuracy</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Loss</b>", secondary_y=True)

    fig.show()

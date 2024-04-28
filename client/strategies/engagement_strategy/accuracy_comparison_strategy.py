from strategies.engagement_strategy.base import EngagementStrategy

class AccuracyComparisonStrategy(EngagementStrategy):
    def __init__(self, data):
        self.data = data
    def calculate_engagement(self):
        if self.data['g_acc'] > self.data['l_acc']:
            return 1
        return 0
        
from django.db import models

class ForecastHistory(models.Model):
    dataset_name   = models.CharField(max_length=255)
    forecast_days  = models.IntegerField(default=30)
    model_used     = models.CharField(max_length=100)
    mae            = models.FloatField(null=True, blank=True)
    rmse           = models.FloatField(null=True, blank=True)
    trend          = models.CharField(max_length=50, default='Unknown')
    total_records  = models.IntegerField(default=0)
    forecast_start = models.CharField(max_length=50, blank=True)
    forecast_end   = models.CharField(max_length=50, blank=True)
    created_at     = models.DateTimeField(auto_now_add=True)
    result_file    = models.CharField(max_length=500, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.dataset_name} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

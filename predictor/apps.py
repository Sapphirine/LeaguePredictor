from django.apps import AppConfig
from django.conf import settings
from threading import RLock

class MyAppConfig(AppConfig):
    name = 'predictor'
    verbose_name = 'League Predictor'

    def ready(self):
        import sys, os

        sys.path.append('{0}/python/build'.format(settings.SPARK_HOME))
        sys.path.append('{0}/python'.format(settings.SPARK_HOME))
        os.environ['SPARK_HOME'] = settings.SPARK_HOME

        from pyspark import SparkContext

        import learn
        self.sc = SparkContext()
        self.model1, self.model2 = learn.main(self.sc)

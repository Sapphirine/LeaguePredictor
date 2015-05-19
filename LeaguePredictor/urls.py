from django.conf.urls import include, url
from django.contrib import admin

urlpatterns = [
    # Examples:
    # url(r'^$', 'LeaguePredictor.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
    url(r'^$', 'predictor.views.index', name = 'index'),
    url(r'^predict/', 'predictor.views.prediction', name = 'prediction'),
]

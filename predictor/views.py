from django.shortcuts import render
from django.http import HttpResponse
from django.apps import apps
from django.conf import settings

from .forms import PredictorForm

import predictor

# def championList():
#     import json, urllib2
#     data = json.loads(urllib2.urlopen('https://na.api.pvp.net/api/lol/static-data/na/v1.2/champion?&api_key={0}'.format(settings.RIOT_API_KEY)).read())['data']
#     champions = {}

#     for champ in data:
#         champions[data[champ]['id']] = data[champ]['name']

#     return champions

def championList():
	import json

	with open('champions') as input:
		data = json.loads(input.read())['data']

	champions = {}

	for champ in data:
		champions[int(data[champ]['key'])] = data[champ]['name']

	return champions

def index(request):
	model1, margin1 = apps.get_app_config('predictor').model1
	model2, margin2 = apps.get_app_config('predictor').model2

	if request.method == 'POST':
		form = PredictorForm(request.POST)
		
		if form.is_valid():
			champions = [int(form.cleaned_data['champ{0}'.format(x)]) for x in range(1, 11)]
			roles = [int(form.cleaned_data['role{0}'.format(x)]) for x in range(1, 11)] if int(form.cleaned_data['useRoles']) else None

			if 'predict' in request.POST:
				from learn import predictWithModel
				x = predictWithModel(model2 if roles else model1, champions, roles = roles)

				conf = abs(x) / (margin2 if roles else margin1)
				arr = ['Low', 'Medium', 'High']
				disp = arr[int(conf) - 1] if conf < 4 else 'Very High'

				return HttpResponse('{0} team is expected to win with {1} confidence ({2:.2f})'.format('Blue' if x > 0 else 'Purple', disp, conf))
			
			elif 'next' in request.POST:
				from learn import bestNextChamp
				nextPick = int(form.cleaned_data['nextPick'])
				x = bestNextChamp(apps.get_app_config('predictor').sc, model2 if roles else model1, nextPick - 1, champions, roles = roles)
				champMap = championList()

				x = [champMap[c] for c in x[0]]

				return HttpResponse('The best champion to pick next is (/is one of) {0}'.format(', '.join(x)))

	return render(request, 'predictor/index.html', { 'form': PredictorForm() })

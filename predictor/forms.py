from django import forms
from django.conf import settings

# def championList():
#     import json, urllib2
#     data = json.loads(urllib2.urlopen('https://na.api.pvp.net/api/lol/static-data/na/v1.2/champion?&api_key={0}'.format(settings.RIOT_API_KEY)).read())['data']
#     champions = []

#     for champ in data:
#         champions.append((data[champ]['id'], data[champ]['name']))

#     champions.sort(key = lambda (x, y): y)
#     return champions

def championList():
    import json

    with open('champions') as input:
        data = json.loads(input.read())['data']

    champions = []

    for champ in data:
        champions.append((int(data[champ]['key']), data[champ]['name']))

    champions.sort(key = lambda (x, y): y)
    return champions

class PredictorForm(forms.Form):
    champs = championList()
    roles = [(1, 'Top'), (0, 'Jungle'), (2, 'Mid'), (3, 'AD Carry'), (4, 'Support')]

    useRoles = forms.TypedChoiceField(label = 'Use Role Info?', choices = [(0, 'No'), (1, 'Yes')])

    champ1  = forms.TypedChoiceField(label = 'Blue Champion 1', choices = champs)
    role1   = forms.TypedChoiceField(label = 'Role', choices = roles)

    champ2  = forms.TypedChoiceField(label = 'Blue Champion 2', choices = champs)
    role2   = forms.TypedChoiceField(label = 'Role', choices = roles)

    champ3  = forms.TypedChoiceField(label = 'Blue Champion 3', choices = champs)
    role3   = forms.TypedChoiceField(label = 'Role', choices = roles)

    champ4  = forms.TypedChoiceField(label = 'Blue Champion 4', choices = champs)
    role4   = forms.TypedChoiceField(label = 'Role', choices = roles)

    champ5  = forms.TypedChoiceField(label = 'Blue Champion 5', choices = champs)
    role5   = forms.TypedChoiceField(label = 'Role', choices = roles)

    champ6  = forms.TypedChoiceField(label = 'Purple Champion 1', choices = champs)
    role6   = forms.TypedChoiceField(label = 'Role', choices = roles)

    champ7  = forms.TypedChoiceField(label = 'Purple Champion 2', choices = champs)
    role7   = forms.TypedChoiceField(label = 'Role', choices = roles)

    champ8  = forms.TypedChoiceField(label = 'Purple Champion 3', choices = champs)
    role8   = forms.TypedChoiceField(label = 'Role', choices = roles)

    champ9  = forms.TypedChoiceField(label = 'Purple Champion 4', choices = champs)
    role9   = forms.TypedChoiceField(label = 'Role', choices = roles)

    champ10 = forms.TypedChoiceField(label = 'Purple Champion 5', choices = champs)
    role10  = forms.TypedChoiceField(label = 'Role', choices = roles)

    nextPick = forms.TypedChoiceField(label = 'Next Pick Number', choices = [(x, x) for x in range(1, 11)])

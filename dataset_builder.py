def m(key):
	with open('saved_state') as input:
		input = input.read().split('\n')
		old_summoners = eval(input[0])
		new_summoners = eval(input[1])
		games = eval(input[2])
		fileNum = int(input[3]) + 1

		print old_summoners, new_summoners, games, fileNum

	old_summoners, new_summoners, games = main(old_summoners, new_summoners, games, 'data/data{0}.json'.format(fileNum), key)

	with open('saved_state', 'w') as output:
		print >> output, old_summoners
		print >> output, new_summoners
		print >> output, games
		print >> output, fileNum
		print >> output, len(games)

def main(old_summoners, new_summoners, games, fname, key):
	import json, urllib2, pprint
	from time import sleep, time

	start = time()
	dataset = { 'matches': [] }
	baseUrl = 'https://na.api.pvp.net/api/lol/na'

	req = 0

	try:
		for i in range(100):
			print req, time() - start

			if len(new_summoners) == 0:
				print 'no new summoners found'
				return

			print i
			url = '{0}/v2.2/matchhistory/{1}?api_key={2}'.format(baseUrl, new_summoners[0], API_KEY)

			old_summoners.append(new_summoners[0])
			new_summoners = new_summoners[1:]
			
			print url
			data = json.loads(urllib2.urlopen(url).read())	
			req += 1	

			for match in data['matches']:
				sleep(1.2)
				if match['queueType'] == 'RANKED_SOLO_5x5':
					matchId = match['matchId']

					if matchId not in games:
						matchData = json.loads(urllib2.urlopen('{0}/v2.2/match/{1}?api_key={2}'.format(baseUrl, match['matchId'], API_KEY)).read())
						req += 1
						dataset['matches'].append(matchData)
						games.append(match['matchId'])

						for summoner in matchData['participantIdentities']:
							summonerId = summoner['player']['summonerId']
							if summonerId not in old_summoners and summonerId not in new_summoners:
								new_summoners.append(summonerId)
				else:
					print match['queueType']
	except Exception as e:
		print e
	finally:
		with open(fname, 'w') as outFile:
			print >> outFile, json.dumps(dataset)

		print 'Finished in {0} minutes.'.format((time() - start) / 60.)
		return old_summoners, new_summoners, games

if __name__ == '__main__':
	from sys import argv

	if len(argv) != 2:
		print 'Usage: > python dataset_builder.py <riot_api_key>'
	else:
		for i in range(20):
			m(argv[1])
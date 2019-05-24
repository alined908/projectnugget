# Project Nugget
Overwatch CV - Get stats from an Overwatch game recording

## 1. Example Frame (from VOD):
![45300](https://user-images.githubusercontent.com/47507106/57599121-88fe4a00-750a-11e9-9265-bb880f68e899.jpg)

## 2. Example CSV Output:
- Example visible in csv/to_csv folder

## 3. Example stats:
![examplestats](https://user-images.githubusercontent.com/47507106/57599467-9c5de500-750b-11e9-8130-c483474d84ea.PNG)
![fightstats](https://user-images.githubusercontent.com/47507106/58194561-ba011c00-7c7a-11e9-8371-67e2fc923983.PNG)

# Instructions
1. Parse the VOD - 'python parse_vod.py' on an Overwatch recording
2. Place images into a folder in vod_data with appropriate format listed in 'Reminders'
3. Run models on images - 'python to_csv.py'
4. Match Statistics - 'python get_map_stats.py'

# TODO
--------
## Major
- Set up database to store statistics

## Minor
- Eliminate (or don't count) rows that happen x seconds before new round.
- Add roundtype to statistics
- Suicides
- Refactor code

## Stretch
- Implement assists tracking

## Reminders
- Format of VOD name should be ex. 01.04.2019+SF+vs+DAL+RIALTO where date format is dd.mm.yyyy
- Clean ult charges (esp. d.va)
- Input roundtype manually

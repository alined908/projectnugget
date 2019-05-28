# Project Nugget
Overwatch CV - Get stats from an Overwatch game recording

## 1. Example CSV Output:
- Example visible in csv/to_csv folder

## 2. Example stats:
![examplestats](https://user-images.githubusercontent.com/47507106/57599467-9c5de500-750b-11e9-8130-c483474d84ea.PNG)
![fightstats](https://user-images.githubusercontent.com/47507106/58194561-ba011c00-7c7a-11e9-8371-67e2fc923983.PNG)

# Prerequisites
* **OBS Settings**
  * ![obssettings](https://user-images.githubusercontent.com/47507106/58447188-80844280-80b8-11e9-8c9a-ee7054c0c113.PNG)

* **Overwatch Settings**
  * Colorblind Options: Enemy UI: Blue, Friendly: Yellow
  * Lobby needs to be reset one time after entering a new map (Blizzard's colorblind bug)
  * Resolution: 1920 x 1080

* **Other**
  * Create these empty folders in `Project Nugget/` --> `/vods`, `/vod_data`, `/data`
  * Python 3.6 +

## Instructions
1. Place .mp4 scrim recordings in `/vods`
1. `python parse_vod.py` ---> Parses vods in `/vods` into images  
1. Move folders with images generated in `/data` into `/vod_data` where format of folder should be ex. `01.04.2019+SF+vs+DAL+RIALTO`  where date format is `dd.mm.yyyy`
    * Delete map loading images (in between lobby and actual map)
    * If map is koth, get image `roundreset.jpg` and replace first image of new round (round 2, 3)
1. `python to_csv.py` ----> Runs models on match images to output csv
1. Clean csv
    * Create a column after 'Opponent' called 'Roundtype' and input roundtype manually (Ex. Attack/Defense/Gardens/Shrine).  If you see a long chain (>6) of same duration, then roundswap happens at the second row of the ones that have the same duration (aka when "unknownhero" appears)
    * Go over D.va's ult charge and correct column if need be to be accurately representing D.va's ult situation
    * Glance over csv and make sure rows make sense
1. `python get_map_stats.py` -- > Outputs match statistics

## TODO
* **Major**
  * Set up database to store statistics

* **Minor**
  * Suicides & Eliminate (or don't count) rows that happen x seconds before new round.

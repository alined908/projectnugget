# Project Nugget
A computer vision project created for SF Shock, a professional esports team in the Overwatch League.  Overwatch has no API to interact with and obtain statistics of practice matches, but hopefully with this, one can record scrims from the spectator POV and extract valuable game information/statistics from the VOD.

## Getting Started
### Github Setup
```
$ git clone https://github.com/alined908/projectnugget.git
$ cd projectnugget
```

### Prerequisites
* **Dependencies**
  * Python 3.6 +

* [**OBS Settings**](misc/obssettings.PNG)

* **Overwatch Settings**
  * Colorblind Options: Enemy UI: Blue, Friendly: Yellow
  * Lobby needs to be reset one time after entering a new map (Blizzard's colorblind bug)
  * Resolution: 1920 x 1080

* **Other**
  * Create these empty folders in `Project Nugget/` --> `/vods`, `/vod_data`, `/data`

### Instructions
1. Place .mp4 scrim recordings in `/vods`
1. `python parse_vod.py` ---> Parses vods in `/vods` into images  
1. Move folders with images generated in `/data` into `/vod_data` where format of folder should be ex. `01.04.2019+SF+vs+DAL+RIALTO`  where date format is `dd.mm.yyyy`
    * Delete map loading images (in between lobby and actual map)
    * If map is koth, get image [**roundreset.jpg**](misc/roundreset.jpg) and replace first image of new round (round 2, 3)
1. `python to_csv.py` ----> Runs models on match images to output csv
1. Clean csv
    * Create a column after 'Opponent' called 'Roundtype' and input roundtype manually (Ex. Attack/Defense/Gardens/Shrine).  If you see a long chain (>6) of same duration, then roundswap happens at the second row of the ones that have the same duration (aka when "unknownhero" appears)
    * Go over d.va's ult charge and correct column if need be to be accurately representing d.va's ult situation
    * Glance over csv and make sure rows make sense
1. `python get_map_stats.py` -- > Outputs match statistics

### Example Outputs
* [**CSV Output**](csv/to_csv/01.04.2019+SF+vs+DAL+RIALTO.csv)
* [**Match Summary**](misc/matchsummary.PNG) **|** [**Fight Statistics**](misc/fightstats.PNG)

### Todo
* **Major**
  * Set up database to store statistics
* **Minor**
  * Suicides & Eliminate (or don't count) rows that happen x seconds before new round.

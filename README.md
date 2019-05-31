# Project Nugget
A computer vision project created for SF Shock, a professional esports team in the Overwatch League.  Overwatch has no API to interact with and obtain statistics of practice matches, but hopefully with this, one can record scrims from the spectator POV and extract valuable game information/statistics from the VOD. This project uses convolutional neural nets that are trained on datasets created by ourselves. Training set images will be available soon.
## Getting Started
### Project Setup
```
$ git clone https://github.com/alined908/projectnugget.git
$ cd projectnugget
$ pip install opencv-python
$ pip install tensorflow
$ pip install keras
$ pip install numpy
$ pip install pandas
$ pip install matplotlib
$ pip install imutils
$ pip install pytesseract
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
  * Create these empty folders in `Project Nugget/` --> `/vods`, `/vod_data`, `/csvs/to_csv`, `/csvs/get_map_stats`
  * Format of vod names should be ex. `01.04.2019+SFS+vs+DAL+RIALTO`  where date is `dd.mm.yyyy`

## Instructions
1. Place .mp4 scrim recording(s) in `/vods` and run `python parse_vod.py`
1. Clean folder(s) of images in `/vod_data`
    * Delete map loading images [**Example 1**](misc/maploading.jpg) , [**Example 2**](misc/maploading2.jpg)
    * If map is koth, get image [**Round Reset**](misc/roundreset.jpg) and replace first image of new round (round 2, 3)
1. `python to_csv.py` ----> Runs models on match images to output csv
1. In `/csvs/to_csv`, Clean csv
    * Create a column after 'Opponent' called 'Roundtype' and input roundtype manually (Ex. Attack/Defense/Gardens/Shrine).
    * Go over hero's ult charge and correct column (usually just D.Va)
1. `python get_map_stats.py` -- > Outputs match statistics in `/csvs/get_map_stats`

## Example Outputs
* [**CSV Output**](csvs/to_csv/01.04.2019%2BSF%2Bvs%2BDAL%2BRIALTO.csv)
* [**General Statistics**](csvs/get_map_stats/01.04.2019%2BSF%2Bvs%2BDAL%2BBUSAN%2BGeneral.csv) **|** [**Compositions**](csvs/get_map_stats/01.04.2019%2BSF%2Bvs%2BDAL%2BBUSAN%2BComps.csv) **|** [**Fight Statistics**](csvs/get_map_stats/01.04.2019%2BSF%2Bvs%2BDAL%2BRIALTO%2BFights.csv)

## Acknowledgements
* Thanks to Farza for his article on Mood and methodology on CNNs
## Todo
* Suicides & Eliminate (or don't count) rows that happen x seconds before new round.
* Implement smart clean csv functionality (ult charge column)

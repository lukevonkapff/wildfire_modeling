# wildfire_modeling

### Data Sources: 

5 public datasets are utilized in this repository. Below are instructions to download each data source:

1. USDA Monitoring Trends in Burn Severity (MTBS): Download using https://mtbs.gov/direct-download -> "Burned Areas Boundaries Dataset". Contains wildfire boundaries in USA from 1984-2024 collected using a number of remote sensing methods. Find metadata at this link: https://data.fs.usda.gov/geodata/edw/edw_resources/meta/S_USA.MTBS_BURN_AREA_BOUNDARY.xml

2. ORNL DAAC Global Fire Atlas (GFA): Download using https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-cms-global-fire-atlas-1642-1. Contains wildfire boundaries globally from 2003-2016 collected using remote sensing. Find metadata at this link: https://d3o6w55j8uz1ro.cloudfront.net/s3-d0f68fa49c8cba12794bb586349f2341/ornl-cumulus-prod-public.s3.us-west-2.amazonaws.com/cms/CMS_Global_Fire_Atlas/comp/CMS_Global_Fire_Atlas.pdf?A-userid=None&Expires=1759940319&Signature=VykoZB6WRKS~tbmEolMjUwtoiFqotCHTxpzlpSLXbC3y3nse2~Tu-FE~gc9Uoe3kPH2~fmtnkc9~0YFMSTHXhLcdiKF62YiUtY4YOt1ERhD2LHuh2FJlBd4-YnRMRb5MgdapogQ1mijhYBmP4ehEl-Qch870e9h994sgyFt~LULYnQE4JjPPBklrGzbEJDUQNGsEbMJ~2w53fnJr6p8932h2grcOk31zwRDhW42FpD6tweiMlxwv1e~oDb-4LMgB5zpJgd2nxacMjkHMQaOOcs61xR5Q6YxnA3IkeKWyqzU8Kfr3e8Nli~xjEpqvtAPo0M0HDybrHJam7p449K3r6w__&Key-Pair-Id=K392G9DSBSGL6V

3. Idaho State University Historical Fires Database: Download using https://giscenter.isu.edu/research/Techpg/HFD/ -> "Download HFD". Contains wildfire boundaries in the western USA (WA, OR, CA, MT, ID, WY, CO, NM, AZ, UT, NV) from 1950-2016 collected in situ.

4. MODIS Terra+Aqua Land Cover Type Yearly L3 Global 500m SIN Grid V061: Download using https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd12q1-061 -> use Earthdata Search to filter from 01/01/2003-12/31/2016. Contains IGBP land cover classifications globally collected using MODIS. This data product is used to classify fires in the MTBS dataset that do not contain a land cover attribute. Find metadata at this link: https://lpdaac.usgs.gov/documents/1409/MCD12_User_Guide_V61.pdf

5. USDA Landfire Existing Vegetation Type (EVT): Download using https://landfire.gov/data/FullExtentDownloads?field_version_target_id=All&field_theme_target_id=All&field_region_id_target_id=4 -> Existing Vegetation Type -> Download CONUS and Alaska for 2016-2024 for a US raster of categorical vegetation types. Additionally, download https://landfire.gov/vegetation/evt -> Attribute Data Table (CSV) for mapping between raster codes and vegetation attributes.

### Code Sources:

The fitting routine uses code from the powerlaw 1.5 python module (https://pypi.org/project/powerlaw/) written by Jeff Alstott (https://github.com/jeffalstott/powerlaw/blob/master/powerlaw.py) as a starting point, with additional distributions and functionalities added on top.

### Environment:
The repo contains an environment.yml file (or requirements.txt). Run the following commands to set up environment for running notebooks.

-> conda env create -f environment.yml
-> conda activate modis_env
-> pip install -e .

Most of the analysis is shown in the notebooks folder, with all functions defined in utils. Some cells in notebooks take a long time to run, so I added lines saving them into the data folder, that way you only have to run them once. The saved geodataframes are too big to be pushed to git.

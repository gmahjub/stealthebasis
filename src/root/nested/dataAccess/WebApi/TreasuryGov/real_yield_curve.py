import xmlschema
from xml.etree import ElementTree
from xmlschema.validators.exceptions import XMLSchemaValidationError
from xmlschema.validators.exceptions import XMLSchemaParseError
from io import StringIO
####################################
from lxml import etree
import lxml.etree as et
# the above two imports are identical
#####################################
import requests
import pysftp
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, DECIMAL, DateTime
from sqlalchemy.sql import text
from sqlalchemy.orm.query import Query
from sqlalchemy.dialects.mysql import insert
import pandas as pd
from datetime import datetime, date
from zipfile import ZipFile
from root.nested.SysOs.os_mux import OSMuxImpl
from root.nested import get_logger
from root.nested.dataAccess.secure_keys_access import SecureKeysAccess

LOGGER = get_logger()
Base = declarative_base()
LOCAL_REAL_YIELD_CURVE_XSD_ZIP_LOC = "/workspace/data/treasurygov/xsd/zips/"
LOCAL_REAL_YIELD_CURVE_XSD_LOC = '/workspace/data/treasurygov/xsd/'

XSD_ZIP_DIR_PATH = 'DailyTreasuryRealYieldCurveRateData.xsd/'
LOCAL_REAL_YIELD_CURVE_XSD = 'DailyTreasuryRealYieldCurveRateData.xsd'
LOCAL_REAL_YIELD_CURVE_XML_LOC = '/workspace/data/treasurygov/xmlfiles/'
LOCAL_TREASURY_GOV_DATA_PATH = "/workspace/data/treasurygov/"
REMOTE_XSD_LOCATION = "https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Documents/" \
                      "DailyTreasuryRealYieldCurveRateData.xsd.zip"
LOCAL_XML_PARSE_ERROR_LOG = "/workspace/data/treasurygov/logs/xml_errors.log"
BATCH_HISTORICAL_XML_FILENAME = "DailyTreasuryRealYieldCurveRateData-Historical.xml"
USTY_REALYC_PYPARSE_TABLE = 'usty_real_yc_pyparse'
DURATION_TO_DBCOL_MAPPING = {60: 'TC_5YEAR',
                             84: 'TC_7YEAR',
                             120: 'TC_10YEAR',
                             240: 'TC_20YEAR',
                             360: 'TC_30YEAR'}


def get_yield_piont_ts_from_csv(query_start_date,
                                query_end_date,
                                points_on_curve=['ALL']):
    """
    The same as get_yield_point_ts_from_db except for 'csv' when we don't have internet connection
    to db.
    :param query_start_date:
    :param query_end_date:
    :param points_on_curve:
    :return:
    """
    return 1


def get_yield_point_ts_from_db(query_start_date,
                               query_end_date,
                               points_on_curve=['ALL']):
    """
    Point on curve is the point on the curve to retrieve yields for in timeseries format.
    :param points_on_curve: this is a list of all the points on curve to include in the result df.
                            Highly discourage leaving it defaulted to 'ALL', query will take long time.
    :param query_start_date: Highly discourage this being NONE, it will get all the data, which will
                 be resource intensive.
    :param query_end_date: Highly discourage this being NONE, it will get all the data, which will
           be resource intensive.
    :return: a dataframe with the timeseries of nominal yields for the point curve between start
             and end date.
    """
    if points_on_curve[0] == 'ALL':
        LOGGER.error("nominal_yield_curve.get_yield_point_ts_from_db(): must set points on curve to something"
                     "other than 'ALL', use subset of curve points!")
        return None
    list_poc_db_col_nm = [DURATION_TO_DBCOL_MAPPING[poc] for poc in points_on_curve]
    list_poc_db_col_nm.insert(0, "NEW_DATE")
    session = get_db_session()
    the_columns = [getattr(DailyRealUsTyGovYieldCurve, poc_db_col_nm) for poc_db_col_nm in list_poc_db_col_nm]
    q = Query(the_columns, session=session)
    from_db = q.filter(DailyRealUsTyGovYieldCurve.NEW_DATE >= str(query_start_date),
                       DailyRealUsTyGovYieldCurve.NEW_DATE <= str(query_end_date)).all()
    df = pd.DataFrame().from_records(from_db)
    df.columns = list_poc_db_col_nm
    df.set_index("NEW_DATE", inplace=True)
    return df


def get_xsd():
    response = requests.get(REMOTE_XSD_LOCATION)
    LOGGER.info("daily_real_yield_curve.get_xsd(): HTTP RESPONSE STATUS CODE " +
                str(response.status_code))
    zip_content = response.content
    zip_timestamp = datetime.now().isoformat()
    local_real_yield_curve_xsd_zip_loc_full_path = OSMuxImpl.get_proper_path(LOCAL_REAL_YIELD_CURVE_XSD_ZIP_LOC)
    with open(local_real_yield_curve_xsd_zip_loc_full_path + "DailyTreasuryRealYieldCurveRateData" +
              str(zip_timestamp) + ".zip", 'wb') as f:
        f.write(zip_content)

    xsd_zip_file = ZipFile(local_real_yield_curve_xsd_zip_loc_full_path + "DailyTreasuryRealYieldCurveRateData" +
                           str(zip_timestamp) + ".zip")
    local_real_yield_curve_xsd_loc = OSMuxImpl.get_proper_path(LOCAL_REAL_YIELD_CURVE_XSD_LOC)
    LOGGER.info("real_yield_curve.get_xsd(): extracting zip file into directory %s",
                local_real_yield_curve_xsd_loc)
    xsd_zip_file.extractall(path=local_real_yield_curve_xsd_loc + "DailyTreasuryRealYieldCurveRateData.xsd/")
    xsd_zip_file.close()


def run_month_insert(month=0, year=0):
    if month is 0:
        month = datetime.now().month
    if year is 0:
        year = datetime.now().year
    MONTH_YIELD_CURVE_URL = "https://data.treasury.gov/feed.svc/DailyTreasuryRealYieldCurveRateData?" \
                            "$filter=month(NEW_DATE)%20eq%20" + str(month) + \
                            "%20and%20year(NEW_DATE)%20eq%20" + str(year)
    url_response = requests.get(url=MONTH_YIELD_CURVE_URL)
    LOGGER.info("daily_real_yeild_curve.write_response_file(): HTTP RESPONSE STATUS CODE " +
                str(url_response.status_code))
    month_tydotgov_real_yields_xml_file = OSMuxImpl.get_proper_path(LOCAL_REAL_YIELD_CURVE_XML_LOC) + \
                                             "DailyTreasuryRealYieldCurveRateData" + \
                                             date(year, month, 1).isoformat() + ".xml"
    with open(month_tydotgov_real_yields_xml_file, 'wb') as f:
        f.write(url_response.content)
    # check empty xml
    entry_count = check_empty_xml(month_tydotgov_real_yields_xml_file)
    if entry_count is 0:
        LOGGER.error("real_yield_curve:run_daily_insert(): the xml file %s is empty, nothing to insert",
                     month_tydotgov_real_yields_xml_file)
        return
    fix_invalid_xml(month_tydotgov_real_yields_xml_file)
    LOGGER.info("real_yield_curve.daily_insert(): ftp'ing into mysql server xml file %s",
                month_tydotgov_real_yields_xml_file)
    sftp_xml(month_tydotgov_real_yields_xml_file)
    LOGGER.info("real_yield_curve.daily_insert(): running parse_xml() on %s ",
                month_tydotgov_real_yields_xml_file)
    df_daily = pyparse_xml(month_tydotgov_real_yields_xml_file)
    yc_entries = df_daily.apply(vectorized_insert_from_pyparse, axis=1)
    session = get_db_session()
    from sqlalchemy.exc import IntegrityError as SqlAlchemyIntegrityError
    from pymysql.err import IntegrityError as PymysqlIntegrityError
    session.flush()
    for yc_entry in yc_entries.iteritems():
        LOGGER.info("real_yield_curve:inserting yield curve entry %s %s", yc_entry[0], yc_entry[1])
        yc_entry_db = session.query(DailyRealUsTyGovYieldCurve).filter_by(Id=yc_entry[1].Id).first()
        if yc_entry_db is None:
            LOGGER.info("real_yield_curve.run_month_insert(%s,%s): entries for date %s do not exist, inserting...",
                        str(year), str(month), yc_entry[1].NEW_DATE)
            session.add(yc_entry[1])
        else:
            LOGGER.info("real_yield_curve.run_month_insert(%s,%s): entries for date %s do exist, updating...",
                        str(year), str(month), yc_entry[1].NEW_DATE)
            yc_entry_db.set_all(yc_entry[1])
    session.commit()


def run_daily_insert():
    TODAY_DATE = datetime.now()
    DAILY_REAL_YIELD_CURVE_URL = "https://data.treasury.gov/feed.svc/DailyTreasuryRealYieldCurveRateData?" \
                                 "$filter=day(NEW_DATE)%20eq%20" + str(TODAY_DATE.day) + \
                                 "%20and%20month(NEW_DATE)%20eq%20" + str(TODAY_DATE.month) + \
                                 "%20and%20year(NEW_DATE)%20eq%20" + str(TODAY_DATE.year)
    url_response = requests.get(url=DAILY_REAL_YIELD_CURVE_URL)
    LOGGER.info("daily_real_yield_curve.write_response_file(): HTTP RESPONSE STATUS CODE " +
                str(url_response.status_code))
    daily_tydotgov_real_yields_xml_file = OSMuxImpl.get_proper_path(LOCAL_REAL_YIELD_CURVE_XML_LOC) + \
                                             "DailyTreasuryRealYieldCurveRateDataToday.xml"
    with open(daily_tydotgov_real_yields_xml_file, 'wb') as f:
        f.write(url_response.content)
    # check empty xml
    entry_count = check_empty_xml(daily_tydotgov_real_yields_xml_file)
    if entry_count is 0:
        LOGGER.error("real_yield_curve:run_daily_insert(): the xml file %s is empty, nothing to insert",
                     daily_tydotgov_real_yields_xml_file)
        return
    fix_invalid_xml(daily_tydotgov_real_yields_xml_file)
    LOGGER.info("real_yield_curve.daily_insert(): ftp'ing into mysql server xml file %s",
                daily_tydotgov_real_yields_xml_file)
    sftp_xml(daily_tydotgov_real_yields_xml_file)
    LOGGER.info("real_yield_curve.daily_insert(): running parse_xml() on %s ",
                daily_tydotgov_real_yields_xml_file)
    df_daily = pyparse_xml(daily_tydotgov_real_yields_xml_file)
    yc_entries = df_daily.apply(vectorized_insert_from_pyparse, axis=1)
    session = get_db_session()
    for yc_entry in yc_entries.iteritems():
        LOGGER.info("real_yield_curve:inserting yield curve entry %s %s", yc_entry[0], yc_entry[1])
        session.add(yc_entry[1])
    session.commit()


def get_db_session():
    mysql_server_ip = SecureKeysAccess.get_mysql_server_ip()
    mysql_username = SecureKeysAccess.get_mysql_server_user()
    mysql_secret = SecureKeysAccess.get_mysql_server_secret()
    LOGGER.info("real_yield_curve.get_db_session(): "
                "creating db engine with pymysql, username %s, secret stays secret, and server_ip %s",
                mysql_username, mysql_server_ip)
    engine = create_engine('mysql+pymysql://' + mysql_username + ':' + mysql_secret + '@' +
                           mysql_server_ip + '/stealbasistrader')
    Session = sessionmaker()
    Session.configure(bind=engine)
    session = Session()
    return session


def get_db_engine():
    mysql_server_ip = SecureKeysAccess.get_mysql_server_ip()
    mysql_username = SecureKeysAccess.get_mysql_server_user()
    mysql_secret = SecureKeysAccess.get_mysql_server_secret()
    engine = create_engine('mysql+pymysql://' + mysql_username + ':' + mysql_secret + '@' +
                           mysql_server_ip + '/stealbasistrader')
    return engine


def sftp_xml(xml_file):
    # sftp the file into the MySQL Uploads directory.
    mysql_server_ip = SecureKeysAccess.get_mysql_server_ip()
    mysql_username = SecureKeysAccess.get_mysql_server_user()
    mysql_secret = SecureKeysAccess.get_mysql_server_secret()
    mysql_windows_uploads_dir = SecureKeysAccess.get_mysql_win_uploads_dir()
    LOGGER.info("real_yield_curve.sftp_xml(): trying sftp with username %s @ host %s",
                mysql_username, mysql_server_ip)
    with pysftp.Connection(host=mysql_server_ip, username=mysql_username, password=mysql_secret) as sftp:
        with sftp.cd(mysql_windows_uploads_dir):
            sftp.put(xml_file)
            server_side_cwd = sftp.getcwd()
            LOGGER.info("real_yield_curve.sftp_xml(): connecting to mysql server: %s ", mysql_server_ip)
            LOGGER.info("real_yield_curve.sftp_xml(): connecting to mysql server as %s ", mysql_username)
            LOGGER.info("real_yield_curve.sftp_xml(): server side cwd is %s ", server_side_cwd)
            if server_side_cwd != mysql_windows_uploads_dir:
                LOGGER.warning(
                    "real_yield_curve.sftp_xml(): server side cwd is DIFFERENT than deposit account name!!")
            else:
                LOGGER.info("real_yield_curve.sftp_xml(): server side cwd is SAME as deposit account name")
            LOGGER.info("real_yield_curve.sftp_xml(): depositing %s ", xml_file)
            LOGGER.info("real_yield_curve.sftp_xml(): depositing in account name %s ", mysql_windows_uploads_dir)
    sftp.close()
    LOGGER.info("real_yield_curve.sftp_xml(): closing sftp connection to mysql server...closed")


def batch_historical_get(no_ftp=False):
    # connect to treasury.gov xml feed.
    all_hist_tydotgov_real_yields_xml_url = "https://data.treasury.gov/feed.svc/DailyTreasuryRealYieldCurveRateData"
    url_response = requests.get(url=all_hist_tydotgov_real_yields_xml_url)
    hist_tydotgov_real_yields_xml_file = OSMuxImpl.get_proper_path(LOCAL_REAL_YIELD_CURVE_XML_LOC) + \
                                            "DailyTreasuryRealYieldCurveRateData-Historical.xml"
    # write out the response to file
    with open(hist_tydotgov_real_yields_xml_file, 'wb') as f:
        f.write(url_response.content)
        LOGGER.info("historical_real_yield_curve.write_response_file(): HTTP Response Status Code " +
                    str(url_response.status_code))
    entry_count = check_empty_xml(hist_tydotgov_real_yields_xml_file)
    if entry_count is 0:
        LOGGER.error("real_yield_curve.batch_historical_get(): historical xml file is empty %s ",
                     hist_tydotgov_real_yields_xml_file)
        no_ftp = True
    if not no_ftp:
        sftp_xml(hist_tydotgov_real_yields_xml_file)
    return hist_tydotgov_real_yields_xml_file


def batch_historical_insert(historical_xml):
    session = get_db_session()
    df_hist = pyparse_xml(historical_xml)
    yc_entries = df_hist.apply(vectorized_insert_from_pyparse, axis=1)
    for yc_entry in yc_entries.iteritems():
        LOGGER.info("real_yield_curve:inserting yield curve entry %s %s", yc_entry[0], yc_entry[1])
        session.add(yc_entry[1])
    session.commit()


def run_historical_insert():
    historical_xml = batch_historical_get(no_ftp=True)
    LOGGER.info("real_yield_curve.run_one_time_historical_insert(): downloaded historical xml file %s",
                historical_xml)
    entry_count = check_empty_xml(xml_file=historical_xml)
    if entry_count == 0:
        LOGGER.error("real_yield_curve.run_historical_insert(): the xml file %s is empty, historical insert failed",
                     historical_xml)
        return
    fix_invalid_xml(xml_file=historical_xml)
    batch_historical_insert(historical_xml=historical_xml)


def run_historical_load_xml_insert():
    """
    This method is a work in progress. We may just use a stored procedure and forget about this from python.
    :return:
    """
    mysql_win_uploads_dir_str = SecureKeysAccess.get_mysql_win_uploads_dir_str()
    server_file_path = mysql_win_uploads_dir_str + BATCH_HISTORICAL_XML_FILENAME
    historical_xml = batch_historical_get(no_ftp=False)
    if check_empty_xml(xml_file=historical_xml) is 0:
        LOGGER.error("real_yield_curve.run_historical_insert(): the xml file %s is empty, historical insert failed!",
                     historical_xml)
        return
    fix_invalid_xml(xml_file=historical_xml)
    session = get_db_session()
    xml_row_identifier = "<m:properties>"
    table_name = "usty_nom_yc_loadxml"

    sql = text("""
        LOAD XML INFILE 'C:/ProgramData/MySQL Server 8.0/Uploads/DailyTreasuryYieldCurveRateData-Historical.xml' 
        REPLACE INTO TABLE stealbasistrader.usty_nom_yc_loadxml ROWS IDENTIFIED BY '<m:properties>'
        """)
    """
    # or - wee more difficult to interpret the command
    employeeGroup = 'Staff'
    employees = connection.execute(
        text('select * from Employees where EmployeeGroup == :group'),
        group=employeeGroup)
    # or - notice the requirement to quote "Staff"
    employees = connection.execute(
        text('select * from Employees where EmployeeGroup == "Staff"'))


    """
    stmt = text(
        "LOAD XML INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/DailyTreasuryYieldCurveRateData-Historical.xml' "
        "REPLACE INTO TABLE stealbasistrader.usty_nom_yc_loadxml ROWS IDENTIFIED BY '<m:properties>'")
    #    stmt = text("LOAD XML INFILE :server_file_path REPLACE INTO TABLE :table_name "
    #                "ROWS IDENTIFIED by :xml_row_identifier")
    # stmt = stmt.bindparams(server_file_path=server_file_path,
    #                       table_name=table_name,
    #                       xml_row_identifier=xml_row_identifier)
    session.execute(sql)
    session.flush()
    session.commit()


def check_empty_xml(xml_file):
    doc = et.parse(xml_file)
    root = doc.getroot()
    entry_counter = 0
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        entry_counter += 1
    if entry_counter == 0:
        LOGGER.error(
            "real_yield_curve.check_valid_xml(): there are no entries in the xml file, empty xml file at %s",
            xml_file)
    return entry_counter


def check_valid_xml(xml_file,
                    xsd_file):
    error_syntax_log = OSMuxImpl.get_proper_path(LOCAL_XML_PARSE_ERROR_LOG)
    with open(xsd_file, 'r') as schema_file:
        schema_to_check = schema_file.read()
    # open and read xml file
    with open(xml_file, 'r') as xml_f:
        xml_to_check = xml_f.read()
    xmlschema_doc = etree.parse(StringIO(schema_to_check))
    xml_schema = etree.XMLSchema(xmlschema_doc)
    # parse xml
    try:
        doc = etree.parse(StringIO(xml_to_check))
        LOGGER.info("real_yield_curve.check_valid_xml(): all is well so far, parsed successfully"
                    " xml file %s", xml_file)
    # check for file IO error
    except IOError as io_error:
        LOGGER.error("real_yield_curve.check_valid_xml(): invalid file, exception is %s", str(io_error))
    # check for XML syntax errors
    except etree.XMLSyntaxError as err:
        LOGGER.error("real_yield_curve.check_xml_valid(): schema validation error %s, see error_schema.log!",
                     str(err))
        with open(error_syntax_log, 'w') as error_log_file:
            error_log_file.write(str(err.error_log))
        quit()
    except:
        LOGGER.error("real_yield_curve.check_xml_valid(): unknown error, quitting...")
        quit()
    # validate against schema
    try:
        xml_schema.assertValid(doc)
        LOGGER.info("real_yield_curve.check_xml_valid(): XML Valid, schema validation ok %s %s", xsd_file, xml_file)
    except etree.DocumentInvalid as err:
        LOGGER.error("real_yield_curve.check_xml_valid(): schema validation error %s %s, see error_schema.log!",
                     xsd_file, xml_file)
        with open(error_syntax_log, 'w') as error_log_file:
            error_log_file.write(str(err.error_log))
        quit()
    except:
        LOGGER.error("real_yield_curve.check_xml_valid(): unknown error, quitting...")
        quit()


def fix_invalid_xml(xml_file):
    with open(xml_file, 'rb+') as f:
        tree = et.parse(f)
        root = tree.getroot()
        for elem in root.getiterator():
            if elem.attrib:
                if "{http://schemas.microsoft.com/ado/2007/08/dataservices/metadata}null" in elem.attrib:
                    del elem.attrib["{http://schemas.microsoft.com/ado/2007/08/dataservices/metadata}null"]
                    LOGGER.info("real_yield_curve.fix_invalid_xml(): removed null from xml file %s "
                                "replace with -1000.0! ", xml_file)
                    elem.text = "-1000.0"
        f.seek(0)
        f.write(et.tostring(tree, encoding='UTF-8', xml_declaration=True))
        f.truncate()


def pyparse_xml(xml_file):
    xsd_file = OSMuxImpl.get_proper_path(LOCAL_REAL_YIELD_CURVE_XSD_LOC) + XSD_ZIP_DIR_PATH + \
               LOCAL_REAL_YIELD_CURVE_XSD
    LOGGER.info("real_yield_curve.parse_xml(): Using xsd schema at %s ", xsd_file)
    try:
        schema = xmlschema.XMLSchema(xsd_file, validation="strict")
        schema.validate(xml_file)
        if not schema.is_valid(xml_file):
            LOGGER.error("real_yield_curve.parse_xml(): %s xml file is NOT validate against schema %s !",
                         xml_file, xsd_file)
            LOGGER.error(
                "real_yield_curve.parse_xml(): is today a weekend day? empty xml files on weekends, check %s",
                xml_file)
        else:
            LOGGER.info("real_yield_curve.parse_xml(): %s xml file is valid against schema %s ! ", xml_file,
                        xsd_file)
    except XMLSchemaValidationError:
        LOGGER.error("real_yield_curve.pyparse_xml(): validation failed!")
    except XMLSchemaParseError:
        LOGGER.error("real_yield_curve.pyparse_xml(): xmlschemaparse error!")

    xt = ElementTree.parse(xml_file)
    root = xt.getroot()
    yc_xml_to_dict = schema.to_dict(xml_file)
    top_level_keys = yc_xml_to_dict.keys()
    entry_yc_xml_list = yc_xml_to_dict['entry']
    return_dict = {}
    for entry_dict in entry_yc_xml_list:
        content_dict = entry_dict['content']
        content_meta_properties = content_dict['m:properties']
        date_of_rate_data = content_meta_properties['d:NEW_DATE']
        sub_return_dict = {}
        for cmp_key in sorted(set(content_meta_properties.keys())):
            act_data_dict = content_meta_properties[cmp_key]
            for add_key in sorted(set(act_data_dict.keys())):
                act_data = act_data_dict[add_key]
                # there is an @m:type key, which is the Entity Data Model data type
                # for example: the d:NEW_DATE key is @m:type Edm.DateTime
                # for example: the d:BC_7EAR key is @m:type is Edm.Double
                # then there is a '$' key, which is the actual data.
                # print("cmp_key", cmp_key)
                # print("add_key", add_key)
                # print("act_data", act_data)
                if sub_return_dict.get(cmp_key, None) is None:
                    value_dict = {add_key: act_data}
                    sub_return_dict[cmp_key] = value_dict
                else:
                    value_dict = sub_return_dict.get(cmp_key)
                    value_dict[add_key] = act_data
                    # we don't need the below since it's a pointer.
                    sub_return_dict[cmp_key] = value_dict
                return_dict[date_of_rate_data['$']] = sub_return_dict
    df = pd.DataFrame.from_dict(data=return_dict, orient='index')
    return df


def vectorized_insert_from_pyparse(row):
    """
    vectorized_insert_from_pyparse(row) takes in a row of a dataframe.
    A yield curve entry object is created and then returned.
    This is the INSERT function we will most typically use.
    :param row:
    :return: yield curve entry object
    <m:properties>
        <d:DailyTreasuryRealYieldCurveRateId m:type="Edm.Int32">4337</d:DailyTreasuryRealYieldCurveRateId>
        <d:NEW_DATE m:type="Edm.DateTime">2020-05-01T00:00:00</d:NEW_DATE>
        <d:TC_5YEAR m:type="Edm.Double">-0.315825</d:TC_5YEAR>
        <d:TC_7YEAR m:type="Edm.Double">-0.399095</d:TC_7YEAR>
        <d:TC_10YEAR m:type="Edm.Double">-0.420718</d:TC_10YEAR>
        <d:TC_20YEAR m:type="Edm.Double">-0.262226</d:TC_20YEAR>
        <d:TC_30YEAR m:type="Edm.Double">-0.121836</d:TC_30YEAR>
      </m:properties>
    """
    # DailyTreasuryRealYieldCurveRateId
    identifier = row.loc['d:DailyTreasuryRealYieldCurveRateId']['$']
    # 10 year govty yield
    ten_yr_yield = row.loc['d:TC_10YEAR']['$']
    # 20 year govty yield
    twenty_yr_yield = row.loc['d:TC_20YEAR']['$']
    # 30 year govty yield
    thirty_yr_yield = row.loc['d:TC_30YEAR']['$']
    # 5 year govty yield
    five_yr_yield = row.loc['d:TC_5YEAR']['$']
    # 7 year govty yield
    seven_yr_yield = row.loc['d:TC_7YEAR']['$']
    # date of yield curve (from treasury.gov)
    new_date = row.loc['d:NEW_DATE']['$']

    yc_entry = DailyRealUsTyGovYieldCurve(Id=identifier,
                                          TC_10YEAR=ten_yr_yield,
                                          TC_20YEAR=twenty_yr_yield,
                                          TC_30YEAR=thirty_yr_yield,
                                          TC_5YEAR=five_yr_yield,
                                          TC_7YEAR=seven_yr_yield,
                                          NEW_DATE=new_date)
    return yc_entry


def insert_from_pyparse(df):
    """
    insert_from_pyparse(df) takes a dataframe that contains yield curve entry for
    one date. Typically, we won't use this method at all, unless for a a single date
    entry.
    """
    # Id
    identifier = df.loc['d:DailyTreasuryRealYieldCurveRateId']['$']
    # 10 year govty yield
    ten_yr_yield = df.loc['d:TC_10YEAR']['$']
    # 20 year govty yield
    twenty_yr_yield = df.loc['d:TC_20YEAR']['$']
    # 30 year govty yield
    thirty_yr_yield = df.loc['d:TC_30YEAR']['$']
    # 5 year govty yield
    five_yr_yield = df.loc['d:TC_5YEAR']['$']
    # 7 year govty yield
    seven_yr_yield = df.loc['d:TC_7YEAR']['$']
    # date of yield curve (from treasury.gov)
    new_date = df.loc['d:NEW_DATE']['$']

    yc_entry = DailyRealUsTyGovYieldCurve(Id=identifier,
                                             TC_10YEAR=ten_yr_yield,
                                             TC_20YEAR=twenty_yr_yield,
                                             TC_30YEAR=thirty_yr_yield,
                                             TC_5YEAR=five_yr_yield,
                                             TC_7YEAR=seven_yr_yield,
                                             NEW_DATE=new_date)
    session_from_yc_entry = yc_entry.Session()
    session_from_yc_entry.add(yc_entry)
    session_from_yc_entry.commit()


class DailyRealUsTyGovYieldCurve(Base):
    __tablename__ = USTY_REALYC_PYPARSE_TABLE
    Id = Column("d:DailyTreasuryRealYieldCurveRateId", Integer, primary_key=True)
    TC_10YEAR = Column("d:TC_10YEAR", DECIMAL)
    TC_20YEAR = Column("d:TC_20YEAR", DECIMAL)
    TC_30YEAR = Column("d:TC_30YEAR", DECIMAL)
    TC_5YEAR = Column("d:TC_5YEAR", DECIMAL)
    TC_7YEAR = Column("d:TC_7YEAR", DECIMAL)
    NEW_DATE = Column("d:NEW_DATE", DateTime)

    def __repr__(self):
        return "<usty_real_yc_pyparse(Id='%s', TC_5YEAR='%s', TC_7YEAR='%s', TC_10YEAR='%s', TC_20YEAR='%s', " \
               "TC_30YEAR='%s', NEW_DATE='%s')>" % \
               (self.Id, self.TC_5YEAR, self.TC_7YEAR, self.TC_10YEAR, self.TC_20YEAR, self.TC_30YEAR, self.NEW_DATE)

    def set_all(self, yc_entry_obj):
        self.Id = yc_entry_obj.Id
        self.TC_5YEAR = yc_entry_obj.TC_5YEAR
        self.TC_7YEAR = yc_entry_obj.TC_7YEAR
        self.TC_10YEAR = yc_entry_obj.TC_10YEAR
        self.TC_20YEAR = yc_entry_obj.TC_20YEAR
        self.TC_30YEAR = yc_entry_obj.TC_30YEAR
        self.NEW_DATE = yc_entry_obj.NEW_DATE


def insert_day_yield_curve():
    """
    Only for testing purposes, do not use this method for production.
    """
    mysql_server_ip = SecureKeysAccess.get_mysql_server_ip()
    mysql_username = SecureKeysAccess.get_mysql_server_user()
    mysql_secret = SecureKeysAccess.get_mysql_server_secret()
    engine = create_engine('mysql+pymysql://' + mysql_username + ':' + mysql_secret + '@' +
                           mysql_server_ip + '/stealbasistrader')
    sm = sessionmaker()
    sm.configure(bind=engine)
    sess = sm()
    yc_entry = DailyRealUsTyGovYieldCurve(Id=100, TC_10YEAR=1.23, TC_20YEAR=0.11)
    sess.add(yc_entry)
    sess.commit()


#df = get_yield_point_ts_from_db(query_start_date="2020-06-01",
#                                query_end_date="2020-06-12",
#                                points_on_curve=[60, 84,120,240,360])
# run_historical_insert() should never be run except 1 time. It has already been run at this point, and the db
# has been pre-populated with all the data available from treasury.gov.
# run_historical_load_xml_insert()

#run_month_insert(5, 2020)
#run_month_insert()

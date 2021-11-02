import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek, monotonically_increasing_id


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_CREDENTIALS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_CREDENTIALS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    """
    # Because I ran this on AWS, this isn't really necessary, but came with the template
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    """
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data', '*', '*', '*')

    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    # drops duplicates
    songs_table = df.select(['song_id', 'title', 'artist_id', 'year',
                             'duration']) \
                    .dropDuplicates()

    # create a temp view of the songs table
    songs_table.createOrReplaceTempView('songs')

    # write songs table to parquet files partitioned by year and artist
    songs_table.write \
           .partitionBy('year', 'artist_id') \
           .mode('overwrite') \
           .parquet(os.path.join(output_data, 'songs', 'songs.parquet'))

    # extract columns to create artists table
    # drops duplicates
    artists_table = df.select('artist_id', 'artist_name', 'artist_location',
                              'artist_latitude', 'artist_longitude') \
                      .withColumnRenamed('artist_name', 'name') \
                      .withColumnRenamed('artist_location', 'location') \
                      .withColumnRenamed('artist_latitude', 'latitude') \
                      .withColumnRenamed('artist_longitude', 'longitude') \
                      .dropDuplicates()

    # create a temp view of the artists table
    artists_table.createOrReplaceTempView('artists')

    # write artists table to parquet files
    artists_table.write \
                 .mode('overwrite') \
                 .parquet(os.path.join(output_data, 'artists', 'artists.parquet'))


def process_log_data(spark, input_data, output_data):
    """
    """

    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data', '*', '*')

    # read log data file
    df = spark.read.json(log_data)

    # filter by actions for song plays
    df = df.where(df.page == 'NextSong')

    # extract columns for users table
    users_table = df.select('userId', 'firstName', 'lastName',
                            'gender', 'level') \
                    .dropDuplicates()

    # write users table to parquet files
    users_table.write \
           .mode('overwrite') \
           .parquet(os.path.join(output_data, 'users', 'users.parquet'))

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: x/1000)
    df = df.withColumn('timestamp', get_timestamp('ts'))

    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x))))
    df = df.withColumn('datetime', get_datetime('timestamp'))

    # extract columns to create time table
    time_table = df.select('datetime') \
                   .withColumn('hour', hour('datetime')) \
                   .withColumn('day', dayofmonth('datetime')) \
                   .withColumn('week', weekofyear('datetime')) \
                   .withColumn('month', month('datetime')) \
                   .withColumn('year', year('datetime')) \
                   .withColumn('weekday', dayofweek('datetime')) \
                   .dropDuplicates()

    # write time table to parquet files partitioned by year and month
    time_table.write \
          .partitionBy('year', 'month') \
          .mode('overwrite') \
          .parquet(os.path.join(output_data, 'time', 'time.parquet'))

    # read in song data to use for songplays table
    song_data = os.path.join(input_data, 'song_data', '*', '*', '*')
    song_df = spark.read.json(song_data)

    # extract columns from joined song and log datasets to create songplays table
    df_joined = df.join(song_df,
                        col('artist') == col('artist_name'),
                        'inner')
    songplays_table = df_joined.select(col('datetime').alias('start_time'),
                                   col('userId').alias('user_id'),
                                   col('level'),
                                   col('song_id'),
                                   col('artist_id'),
                                   col('sessionId').alias('session_id'),
                                   col('location'),
                                   col('userAgent').alias('user_agent'),
                                   year(col('datetime')).alias('year'),
                                   month(col('datetime')).alias('month')) \
                            .withColumn('songplay_id', monotonically_increasing_id())

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write \
               .partitionBy('year', 'month') \
               .mode('overwrite') \
               .parquet(os.path.join(output_data, 'songplays', 'songplays.parquet'))


def main():
    spark = create_spark_session()

    # to test locally, use these variables instead
    # input_data = 'data'
    # output_data = 'output'

    input_data = "s3a://udacity-dend/"
    output_data = ""

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()

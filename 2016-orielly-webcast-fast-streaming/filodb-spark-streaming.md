## Fast and Simplified Streaming,
## Ad-Hoc and Batch Analytics with 
## FiloDB and Spark Streaming

### [Evan Chan](https://twitter.com/Evanfchan) and [Helena Edelson](https://twitter.com/helenaedelson)
### March 2016

---

## Evan Chan

<center>
![](home.jpg)
</center>

- Distinguished Engineer, [Tuplejump](http://www.tuplejump.com)
- [@evanfchan](https://twitter.com/Evanfchan)
- [`http://velvia.github.io`](http://velvia.github.io)
- User and contributor to Spark since 0.9, Cassandra since 0.6
- Co-creator and maintainer of [Spark Job Server](http://github.com/spark-jobserver/spark-jobserver)

---

## Helena Edelson

- VP of Product Engineering, [Tuplejump](http://www.tuplejump.com)
- Committer: Kafka Connect Cassandra, Spark Cassandra Connector
- Contributor to Akka (2 new features in Akka Cluster), Spring Integration
- Speaker @ Spark Summit, Kafka Summit, Strata, QCon, Scala Days, Scala World, Philly ETE 
- [@helenaedelson](https://twitter.com/helenaedelson)
- [`http://github.com/helena`](http://github.com/helena)

---

## Tuplejump

[Tuplejump](http://tuplejump.com) is a big data technology leader providing solutions and development partnership.

---

## ![](images/tuplejump_circle-sm.png) [Tuplejump](http://tuplejump.com) Open Source: on GitHub

* [FiloDB](http://github.com/tuplejump/FiloDB) - Subject of today's talk
* [Kafka Connect Cassandra](http://github.com/tuplejump/kafka-connector) - Kafka-Cassandra Source and Sink
* [Calliope](http://tuplejump.github.io/calliope/) - The first Spark Cassandra integration
* [Stargate](http://tuplejump.github.io/stargate/) - Lucene indexer for Cassandra
* [SnackFS](https://github.com/tuplejump/snackfs) - HDFS for Cassandra

---

## Tuplejump Consulting & Development

<center>
![](images/tj-consulting.png)
</center>

---

<center>
**Tuplejump Blender** 
<small>Builds unified datasets for fast querying, streaming and batch sources, ML and Analytics</small>
</center>
<center>
![](images/tj-blender.png) 
</center>

---

## Topics

* Modern streaming architectures and batch/ad-hoc architectures
* Precise and scalable streaming ingestion using Kafka, Akka, Spark Streaming, Cassandra, and FiloDB
* How a unified streaming + batch stack can lower your TCO
* What FiloDB is and how it enables fast analytics with competitive storage cost
* Data Warehousing with Spark, Cassandra, and FiloDB
* Time series / event data / geospatial examples
* Machine learning using Spark MLLib + Cassandra/FiloDB
* Integrating streaming and historical data analysis

---

## The Problem Domain
Build scalable, adaptable, self-healing, distributed data processing systems for
 
- Massive amounts of data 
- Disparate sources and schemas
- Differing data structures
- Complex analytics and learning tasks
- To run as large-scale clustered dataflows
- 24-7 Uptime
- Globally clustered deployments
- No data loss

---

## Delivering Meaning

- Deliver meaning in sec/sub-sec latency
- Billions of events per second
- Low latency real time stream processing
- Higher latency batch processing
- Aggregation of global data from the stream

---

## While We Monitor, Predict & Proactively Handle

- Massive event spikes & bursty traffic
- Fast producers / slow consumers
- Network partitioning & out of sync systems 
- DC down 
- Not DDOS'ing ourselves from fast streams
- No data loss when auto-scaling down

---

## And stay within our 
<center>
AWS<br/>
OpenStack<br/>
Rackspace...<br/> 

budget</center>

---

<center>I need fast access to historical data on the fly for predictive modeling 
        with real time data from the stream 
</center>
<center>
![](images/predictive.jpeg)
</center>

---

## Only, It's Not A Stream It's A Flood

Netflix

- 100 billion events per day
- 1-2 million events per second at peak

Linkedin

- 500 billion write events per day
- 2.5 trillion read events per day
- 4.5 million events per second at peak with Kafka
- Petabytes of streaming data

Confidential

- 700 TB global ingested data (pre-filtered)

---

## Lambda Architecture

*A data-processing architecture designed to handle massive quantities 
of data by taking advantage of both batch and stream processing methods.*

---

## Lambda Architecture

<center>
![Lamba Architecture](lambda-architecture-2-800.jpg)
</center>

<center>
(https://www.mapr.com/developercentral/lambda-architecture)
</center>

---

## Lambda: Multiple Systems
 
- Can be a very complex pipeline
- Many moving parts - KV store, real time, Batch technologies plus ETL...
- Running similar code in two places
- Reconcile queries against two different places
- Complicated logic changes across multiple systems (code)

---

## Challenges with Multiple Systems
 
- Overly-complicated Ops
- Complicated fault tolerance across all systems (immutability)
- Performance tuning & monitoring on multiple systems
- High TCO

---

## Which Translates To

<center>
![](images/architectyr.jpg)
</center>

---

## Why Separate Streaming & Batch Flows

- Evolution Or Just Addition?
- Or Just Technical Debt?

---

## Challenge Assumptions

<center>
*"Ingest an immutable sequence of records is captured and fed into<br/> 
a batch processing system<br/> 
and a stream processing system<br/> 
in parallel"*
</center>

---

## Are Batch and Streaming Systems Fundamentally Different?

<center>No.</center>

---

## Do We Really Need 

- A Batch Analytics System
- An ETL System
- A Streaming Analytics System

<br/>
<center>*Not any more.*</center>

---

## Why Dual Analytics Systems

Supporting code, machines, staff, monitoring and running services for multiple clustered analytics systems?

- Performing analytical computations & queries in dual systems
- Spaghetti Architecture for Data Flows
- Duplicated Code
- One Busy Network

---

## A Unified Streaming Architecture
<center>
Everything On The Streaming Platform</center>
<br/>

- Scala / Spark Streaming
- Mesos
- Akka
- Cassandra
- Kafka

---

## Spark Streaming

<center>
![](spark-streaming.png)
</center>

- One runtime for streaming and batch processing
- Join streaming and static data sets 
- No code duplication
- Easy Kafka stream integration
- Easy, flexible data ingestion from disparate sources to disparate sinks
- Easy to reconcile queries against multiple sources
- Easy integration of KV durable storage

--

## <span class="cassred">Cassandra</span>

<center>
![](cassandra.jpg)
</center>

- Horizontally scalable
- Multi-Region / Multi-Datacenter
- Always On - Survive regional outages
- Extremely fast writes: - perfect for ingestion of real time / machine data
- Very flexible data modelling (lists, sets, custom data types)
- Easy to operate
- Best of breed storage technology, huge community
- **BUT: Simple queries only**
- **OLTP-oriented**

---

## Akka

High performance concurrency framework for Scala and Java 
Fault Tolerance
Asynchronous messaging and data processing 
Parallelization
Location Transparency
Local / Remote Routing 
Akka: Cluster / Persistence / Streams

---

## Immutable Raw Data From Kafka Stream
<center>Replay / reprocessin: for fault tolerance, logic changes..</center>

```scala
     val stream = KafkaUtils.createDirectStream(...) .map(RawWeatherData(_))
     stream
       .foreachRDD(_.toDF.write.format("filodb.spark")
       .option(rawDataKeyspace, rawDataTable))
```
## Pre-Aggregate Data In The Stream 
<center>For fast querying and further aggregation later</center>

```scala
     stream.map(hour => 
       (hour.wsid, hour.year, hour.month, hour.day, hour.oneHourPrecip)
     ).saveToCassandra(CassandraKeyspace, CassandraTableDailyPrecip)  
```

---

### Reading Data Back From Cassandra
#### Compute isolation in Akka Actor

```scala

    class TemperatureActor(sc: SparkContext, settings: Settings) extends AggregationActor {   
      import settings._
      import akka.pattern.pipe
       
      def receive: Actor.Receive = {     
        case e: GetMonthlyHiLowTemperature => highLow(e, sender)   
      }     
      
      def highLow(e: GetMonthlyHiLowTemperature, requester: ActorRef): Unit =     
        sc.cassandraTable[DailyTemperature](timeseriesKeyspace, dailyTempAggregTable)       
          .where("wsid = ? AND year = ? AND month = ?", e.wsid, e.year, e.month)
          .collectAsync()       
          .map(MonthlyTemperature(_, e.wsid, e.year, e.month)) pipeTo requester
    }
    
```

---

## Spark Streaming, MLLib
## Kafka, Cassandra

```scala
val ssc = new StreamingContext(sparkConf, Seconds(5) 
val testData = ssc.cassandraTable[String](keyspace,table)
  .map(LabeledPoint.parse)
      
val trainingStream = KafkaUtils.createDirectStream[_,_,_,_](..)
    .map(transformFunc)
    .map(LabeledPoint.parse)
    
trainingStream.saveToCassandra("ml_training_keyspace", "raw_training_data")  
    
 val model = new StreamingLinearRegressionWithSGD()   
  .setInitialWeights(Vectors.dense(weights))   
  .trainOn(trainingStream)
     
model
  .predictOnValues(testData.map(lp => (lp.label, lp.features)))
  .saveToCassandra("ml_predictions_keyspace", "predictions")

```

---

## Tuplejump OSS Roadmap

- [Gearpump](http://www.gearpump.io/overview.html) gearpump-external-filodb
- Kafka Connect FiloDB

---

## What's Missing? One Pipeline For Fast + Big Data

![](one-pipeline.mermaid.png)
<!-- .element: class="mermaid" -->

--

## Ad-Hoc and ML with Spark

DataFrames support in Spark Cassandra Connector:

<p>
```scala
val sqlContext = new org.apache.spark.sql.SQLContext(sc)

val df = sqlContext.read
                   .format("org.apache.spark.sql.cassandra")
                   .option("table", "gdelt")
                   .option("keyspace", "test").load()
df.registerTempTable("gdelt")
sqlContext.cacheTable("gdelt")
sqlContext.sql("SELECT count(monthyear) FROM gdelt").show()
```

--

## Spark provides the missing fast, deep analytics piece of <span class="cassred">Cassandra</span>!

### ...tying together fast event ingestion and rich deep analytics!

---

## What about Parquet?

- Good for static data
- Problem: Parquet is read-optimized, not easy to use for writes
    + Cannot support idempotent writes
    + Optimized for writing very large chunks, not small updates
    + Not suitable for time series, IoT, etc.
    + Often needs multiple passes of jobs for compaction of small files, deduplication, etc.

&nbsp;
<p>
People really want a database-like abstraction, not a file format!

--

## What are my storage needs?

- Non-persistent / in-memory: concurrent viewers
- Short term: latest trends
- Longer term: raw event and aggregate storage
- ML Models, predictions, scored data

--

## Spark RDDs

- Immutable, cache in memory and/or on disk
- Spark Streaming: UpdateStateByKey
- IndexedRDD - can update bits of data
- Snapshotting for recovery

--

## Using Cassandra for Short Term Storage

|          | 1020s | 1010s | 1000s |
| -------- | ----- | ----- | ----- |
| Bus A    | Speed, GPS |   |      |
| Bus B    |       |        |      |
| Bus C    |       |        |      |

- Primary key = (Bus UUID, timestamp)
- Easy queries: location and speed of single bus for a range of time
- Can also query most recent location + speed of all buses (slower)

--

## Using Cassandra for Longer-Term Event Storage / ML?

- Storage efficiency and scan speeds for reading large volumes of data (for complex analytics, ML) become important concerns
- Regular Cassandra CQL tables are not very good at either storage efficiency or scan speeds
- Have to be a bit creative with how you store data in Cassandra  :)

---

> All hard work leads to profit, but mere talk leads to poverty.<br>
> - Proverbs 14:23

---

## Introducing <span class="golden">FiloDB</span>

<center>
A distributed, versioned, columnar analytics database.<br>
*Built for Streaming.*
</center>

<p>&nbsp;<p>
<center>
[github.com/tuplejump/FiloDB](http://github.com/tuplejump/FiloDB)
</center>

--

## Fast Analytics Storage

- Scan speeds competitive with Apache Parquet
  + Up to 200x faster scan speeds than with Cassandra 2.x
- Flexible filtering along two dimensions
  + Much more efficient and flexible partition key filtering
- Efficient columnar storage, up to 27x more efficient than Cassandra 2.x

NOTE: 200x is just based on columnar storage + projection pushdown - no filtering on sort or partition keys, and no caching done yet.

--

## Comparing Storage Costs and Query Speeds

<center>
![](storage_cost_vs_query.png)
</center>

<center>
[https://www.oreilly.com/ideas/apache-cassandra-for-analytics-a-performance-and-storage-analysis](https://www.oreilly.com/ideas/apache-cassandra-for-analytics-a-performance-and-storage-analysis)
</center>

--

## Robust Distributed Storage

Apache Cassandra as the rock-solid storage engine.  Scale out with no SPOF.  Cross-datacenter replication.
Proven storage and database technology.

--

## Cassandra-Like Data Model

<table>
  <tr>
    <td></td>
    <td colspan="2">Column A</td>
    <td colspan="2">Column B</td>
  </tr>
  <tr>
    <td>Partition key 1</td>
    <td>Segment 1</td>
    <td>Segment 2</td>
    <td>Segment 1</td>
    <td>Segment 2</td>
  </tr>
  <tr>
    <td>Partition key 2</td>
    <td>Segment 1</td>
    <td>Segment 2</td>
    <td>Segment 1</td>
    <td>Segment 2</td>
  </tr>
</table>
&nbsp;<p>

- **partition keys** - distributes data around a cluster, and allows for fine grained and flexible filtering
- **segment keys** - do range scans within a partition, e.g. by time slice
- primary key based ingestion and updates

--

## Flexible Filtering

Unlike Cassandra, FiloDB offers very flexible and efficient filtering on partition keys.  Partial key matches, fast IN queries on any part of the partition key.  
<br/>
*No need to write multiple tables to work around answering different queries.*

--

## Spark SQL Queries!

```sql
CREATE TABLE gdelt USING filodb.spark OPTIONS (dataset "gdelt");

SELECT Actor1Name, Actor2Name, AvgTone FROM gdelt ORDER BY AvgTone DESC LIMIT 15;

INSERT INTO gdelt SELECT * FROM NewMonthData;
```

- Read to and write from Spark Dataframes
- Append/merge to FiloDB table from Spark Streaming
- Use Tableau or any other JDBC tool

--

## What's in the name?

<center>
![Filo dough](Filo.jpg)
</center>

Rich sweet layers of distributed, versioned database goodness

--

## SMACK stack for all your Analytics

![](simple-architecture.mermaid.png)
<!-- .element: class="mermaid" -->

- Regular Cassandra tables for highly concurrent, aggregate / key-value lookups (dashboards)
- FiloDB + C* + Spark for efficient long term event storage
  - Ad hoc / SQL / BI
  - Data source for MLLib / building models
  - Data storage for classified / predicted / scored data

--

![](FiloDB_Cass_Together.001.jpg)

--

![](FiloDB_Cass_Together.002.jpg)

--

## Being Productionized as we speak...

- One enterprise with many TB of financial and reporting data is moving their data warehouse to FiloDB + Cassandra + Spark
- Another startup uses FiloDB as event storage, feeds the events into Spark MLlib, scores incoming data, then stores the results back in FiloDB for low-latency use cases

---

## Data Warehousing with <span class="golden">FiloDB</span>

--

## Scenarios

- BI Reporting, concurrency + seconds latency
- Ad-hoc queries
- Needing to do JOINs with fact tables + dimension tables
  + Slowly changing dim tables / hard to denormalize
- Need to work with legacy BI tools

--

## Real-world DW Architecture Stack

![](data-warehouse.mermaid.png)
<!-- .element: class="mermaid" -->

<center>
Efficient columnar storage + filtering = low latency BI
</center>

--

## Modeling Fact Tables for FiloDB

- Single partition queries are really fast and take up only one thread
  + Given the following two partition key columns: entity_number, year_month
  + `WHERE entity_number = '0453' AND year_month = '2014 December'`
    * Exact match for partition key is pushed down as one partition
- Consider the partition key carefully

--

## Cassandra often requires multiple tables

What about the queries that do not translate to one partition?  Cassandra has many restrictions on partition key filtering (as of 2.x).

- Table 1:  partition key = `(entity_number, year_month)`
  + Can push down: `WHERE entity_number = NN AND year_month IN ('2014 Jan', '2014 Feb')` as well as equals
- Table 2:  partition key = `(year_month, entity_number)`
  + Can push down: `WHERE year_month = YYMM AND entity_number IN (123, 456)` as well as equals

IN clause must be the last column to be pushed down. Two tables are needed just for efficient IN queries on either entity_number or year_month.

--

## FiloDB Flexible Partition Filters = WIN

With ONE table, FiloDB offers FAST, arbitrary partition key filtering.  All of the below are pushed down:

- `WHERE year_month IN ('2014 Jan', '2014 Feb')` (all entities)
- `WHERE entity_number = 146`  (all year months)
- Any combo of =, IN

<p>
<center><large>
Space savings: 27 `*` 2 = **54x**
</large>
</center>

--

## Multi-Table JOINs with just Cassandra

<center>
![](Four-C-Table-Spark-Join.png)
</center>

--

## Sub-second Multi-Table JOINs with FiloDB

<center>
![](Four-Table-Spark-Join.png)
</center>

--

## Sub-second Multi-Table JOINs with FiloDB

- Four tables, all of them single-partition queries
- Two tables were switched from regular Cassandra tables to FiloDB tables.  40-60 columns each, ~60k items in partition.
- Scan times went down from 5-6 seconds to < 250ms

For more details, please see blog post: XXXX

---

## Scalable Time-Series / Event Storage with <span class="golden">FiloDB</span>

--

## Designed for Streaming

- New rows appended via Spark Streaming or Kafka
- Writes are *idempotent* - easy **exactly once** ingestion
- Converted to columnar chunks on ingest and stored in C*
- FiloDB keeps your data sorted as it is being ingested

--

## Spark Streaming -> FiloDB

```scala
    val ratingsStream = KafkaUtils.createDirectStream[String, String, StringDecoder, StringDecoder](ssc, kafkaParams, topics)
    ratingsStream.foreachRDD {
      (message: RDD[(String, String)], batchTime: Time) => {
        val df = message.map(_._2.split(",")).map(rating => Rating(rating(0).trim.toInt, rating(1).trim.toInt, rating(2).trim.toInt)).
          toDF("fromuserid", "touserid", "rating")
      
        // add the batch time to the DataFrame
        val dfWithBatchTime = df.withColumn("batch_time", org.apache.spark.sql.functions.lit(batchTime.milliseconds))
      
        // save the DataFrame to FiloDB
        dfWithBatchTime.write.format("filodb.spark")
          .option("dataset", "ratings")
          .save()
      }
    }
```
One-line change to write to FiloDB vs Cassandra

--

## Modeling Time Series with FiloDB

| Entity  | Time1 | Time2 |
| ------- | ----- | ----- |
| US-0123 | d1    | d2    |
| NZ-9495 | d1    | d2    |

&nbsp;<p>
Model your time series with FiloDB similarly to Cassandra:

- **Segment key**: `:timeslice timestamp 4h`
- **Partition Keys**:
  - Event/machine UUID (smaller # of events)
  - Event/machine UUID hash + time period
* **Row Keys**: Timestamp, machine UUID

FiloDB keeps data sorted while stored in efficient columnar storage.

--

## Modeling the NYC Taxi Dataset

The public [NYC Taxi Dataset](http://www.andresmh.com/nyctaxitrips/) contains telemetry (pickup, dropoff locations, times) info on millions of taxi rides in NYC.

| Medallion prefix  | 1/1 - 1/6   |  1/7 - 1/12  |
| ----------------- | ----------- | ------------ |
| AA             | records     |  records     |
| AB             | records     |  records     |

* Partition key - `:stringPrefix medallion 2` - hash multiple drivers trips into ~300 partitions
* Segment key - `:timeslice pickup_datetime 6d`

Allows for easy filtering by individual drivers, and slicing by time.

--

# DEMO TIME

### New York City Taxi Data Demo (Spark Notebook)

To follow along:  https://github.com/tuplejump/FiloDB/blob/master/doc/FiloDB_Taxi_Geo_demo.snb

--

<iframe src="http://localhost:9000/notebooks/viz/FiloDB%20Taxi%20Geo%20Demo.snb" height="800" width="1200" />

---

## Fast, Updatable In-Memory
## Columnar Storage

![](streaming-in-memory-filodb.mermaid.png)
<!-- .element: class="mermaid" -->

- Unlike RDDs and DataFrames, FiloDB can ingest new data, and still be fast
- Unlike RDDs, FiloDB can filter in multiple ways, no need for entire table scan
- FAIR scheduler + sub-second latencies => web speed queries

--

## 700 Queries Per Second in Apache Spark!

- Even for datasets with 15 million rows!
- Using FiloDB's `InMemoryColumnStore`, single host / MBP, 5GB RAM
- SQL to DataFrame caching

For more details, see [this blog post](http://velvia.github.io/Spark-Concurrent-Fast-Queries/).

---

## Machine Learning with Spark, Cassandra, and FiloDB

--

## Building a static model of NYC Taxi Trips

- Predict time to get to destination based on pickup point, time of day, other vars
- Need to read all data (full table scan)

--

## Dynamic models are better than static models

- Everything changes!
- Continuously refine model based on recent streaming data + historical data + existing model

--

## The FiloDB Advantage for ML

- Able to update dynamic models based on massive data flow/updates
- More data -> better models!
- Can store scored raw data / predictions back in FiloDB
  + for fast user queries

--

![](FiloDB_Cass_Together.002.jpg)

---

## Integrating Historical and Streaming Workloads

---

## FiloDB - Roadmap

Your input is appreciated!

* Productionization and automated stress testing
* Kafka input API / connector (without needing Spark)
* In-memory caching for significant query speedup
* True columnar querying and execution, using late materialization and vectorization techniques.  GPU/SIMD.
* Projections.  Often-repeated queries can be sped up significantly with projections.

---

## You can help!

- Send me your use cases for fast big data analysis on Spark and Cassandra
    + Especially IoT, Event, Time-Series
    + What is your data model?
- Email if you want to contribute

---

## Thanks For Attending!
 
- [@helenaedelson](https://twitter.com/helenaedelson)
- [@evanfchan](https://twitter.com/Evanfchan)
- [@tuplejump](https://twitter.com/tuplejump) 

---

# EXTRA SLIDES

---

## <span class="golden">FiloDB</span> - How?

--

## Multiple ways to Accelerate Queries

* Columnar projection - read fewer columns, saves I/O
* Partition key filtering - read less data
* Sort key / PK filtering - read from subset of keys
  - Possible because FiloDB keeps data sorted
* Versioning - write to multiple versions, read from the one you choose

--

## Cassandra CQL vs Columnar Layout

Cassandra stores CQL tables row-major, each row spans multiple cells:

| PartitionKey | 01:first | 01:last | 01:age | 02:first | 02:last | 02:age |
| :----------- | :------- | :------ | -----: | :------- | :------ | -----: |
| Sales        | Bob      | Jones   | 34     | Susan    | O'Connor | 40    |
| Engineering  | Dilbert  | P       | ?      | Dogbert  | Dog     |  1     |

&nbsp;<p>
Columnar layouts are column-major:

| PartitionKey | first  |  last  |  age  |
| :----------- | ------ | ------ | ----- |
| Sales        | Bob, Susan | Jones, O'Connor | 34, 40 |
| Engineering  | Dilbert, Dogbert | P, Dog    | ?, 1   |

--

## FiloDB Cassandra Schema

```sql
CREATE TABLE filodb.gdelt_chunks (
    partition text,
    version int,
    columnname text,
    segmentid blob,
    chunkid int,
    data blob,
    PRIMARY KEY ((partition, version), columnname, segmentid, chunkid)
) WITH CLUSTERING ORDER BY (columnname ASC, segmentid ASC, chunkid ASC)
```

--

## FiloDB Architecture

<center>
![](http://velvia.github.io/images/filodb_architecture.png)
</center>

ColumnStore API - currently Cassandra and InMemory, you can implement other backends - ElasticSearch?  etc.


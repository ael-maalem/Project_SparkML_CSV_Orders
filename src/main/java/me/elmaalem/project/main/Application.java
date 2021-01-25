package me.elmaalem.project.main;

import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Application {
    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .appName("Spring Boot App with Spark SQL")
                .master("local[*]")
                .getOrCreate();

        Dataset<Row> csvDF = spark.read()
                .option("header","true")
                .option("treatEmptyValuesAsNulls", "true")
                .option("inferSchema", "true")
                .option("mode","DROPMALFORMED")
                .option("dateFormat", "MM-dd-yyyy")
                .option("delimiter",",")
                .csv("src/main/resources/Orders.csv")
                .select("quantity","sales","profit","unitPrice");

        // Assembling Columns into Features
        System.out.println("******************* Assembler :*******************");
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(csvDF.columns())
                .setOutputCol("features");

        Dataset<Row> orders = assembler.setHandleInvalid("skip").transform(csvDF);
        orders.foreach((ForeachFunction<Row>) row-> System.out.println(row));
        System.out.println("End : ******************* Assembler :*******************");

        // Initialize k clusters with random starting positions
        System.out.println("******************* K-Means :*******************");
        KMeans kMeans = new KMeans()
                .setK(2)
                .setSeed(1L)
                .setFeaturesCol("features");

        // Trains a k-means model
        KMeansModel model = kMeans.fit(orders);
        Dataset<Row> predictions = model.transform(orders);

        predictions.foreach((ForeachFunction<Row>) row-> System.out.println(row));
        System.out.println("End : ******************* K-Means :*******************");

        // Evaluating Cluster Quality
        System.out.println("******************* Evaluating Cluster Quality :*******************");
        ClusteringEvaluator evaluator = new ClusteringEvaluator().setFeaturesCol("features");

        // Compute the mean Silhouette Score [-1,1:perfect]
        System.out.print("******* Silhouette Score : ");
        System.out.println(evaluator.evaluate(predictions));

        // Calculated centers of the clusters
        System.out.println("******* Center of the clusters ********");
        Vector[] centers = model.clusterCenters();
        for (Vector center : centers) {
            System.out.println(center);
        }
        System.out.println("End : ******************* Evaluating Cluster Quality :*******************");
    }
}

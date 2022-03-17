
-- 1. Create a database called house_price_regression.
CREATE DATABASE if not exists house_price_regression;

use house_price_regression;

-- 2. Create a table house_price_data with the same columns as given in the csv file. Please make sure you use the correct data types for the columns.
CREATE TABLE IF NOT EXISTS house_price_data (
id bigint,
date varchar(20),
bedrooms INT,
bathrooms float,
sqft_living int,
sqft_lot int,
floors int,
waterfront int,
view int,
condt int,
grade int,
sqft_above int,
sqft_basement int,
yr_built int,
yr_renovated int,
zipcode int,
latitude float,
longitude float,
sqft_living15 int,
sqft_lot15 int,
price int
);
-- table has no primary key as the id contains several duplicates which could/will be removed later

-- 3. Import the data from the csv file into the table.
SHOW VARIABLES LIKE 'local_infile';

SET GLOBAL local_infile = 1;

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/regression_data.csv' 
INTO TABLE house_price_data 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n';

-- 4. Select all the data from table house_price_data to check if the data was imported correctly
select * from house_price_data;

-- 5. Use the alter table command to drop the column date from the database, as we would not use it in the analysis with SQL. 
-- Select all the data from the table to verify if the command worked. Limit your returned results to 10.
alter table house_price_data drop column datum;
select * from house_price_data
limit 10;

-- 6. Use sql query to find how many rows of data you have.
select count(*) amount_rows from house_price_data;

-- 7. Now we will try to find the unique values in some of the categorical columns:
-- What are the unique values in the column bedrooms?
select distinct(bedrooms) from house_price_data order by bedrooms;

-- What are the unique values in the column bathrooms?
select distinct(bathrooms) from house_price_data order by bathrooms;

-- What are the unique values in the column floors?
select distinct(floors) from house_price_data order by floors;

-- What are the unique values in the column condition?
select distinct(condt) from house_price_data order by condt;

-- What are the unique values in the column grade?
select distinct(grade) from house_price_data order by grade;

-- 8. Arrange the data in a decreasing order by the price of the house. Return only the IDs of the top 10 most expensive houses in your data.
select id from (select id, price from house_price_data order by price desc limit 10) as alias1;

select id, price from house_price_data order by price desc limit 10;

-- 9. What is the average price of all the properties in your data?
select avg(price) as "average price" from house_price_data;

-- 10. In this exercise we will use simple group by to check the properties of some of the categorical variables in our data:
-- What is the average price of the houses grouped by bedrooms? The returned result should have only two columns, bedrooms and Average of the prices.
-- Use an alias to change the name of the second column.
select bedrooms, avg(price) as "average price" from house_price_data
group by bedrooms
order by bedrooms;

-- What is the average sqft_living of the houses grouped by bedrooms? The returned result should have only two columns,
-- bedrooms and Average of the sqft_living. Use an alias to change the name of the second column.
select bedrooms, avg(sqft_living) as "average squarefoot living" from house_price_data
group by bedrooms
order by bedrooms;

-- What is the average price of the houses with a waterfront and without a waterfront? The returned result should have only two columns,
-- waterfront and Average of the prices. Use an alias to change the name of the second column.
select 
	case
		when waterfront = 0 then "no"
        else "yes"
	end as "waterfront",
round(avg(price),2) as "average price"
from house_price_data
group by waterfront;

-- Is there any correlation between the columns condition and grade? You can analyse this by grouping the data by one of the variables
-- and then aggregating the results of the other column. Visually check if there is a positive correlation or negative correlation
-- or no correlation between the variables.

select condt, avg(grade) from house_price_data
group by condt
order by condt;

-- no (strict and continous) correlation

-- 11. One of the customers is only interested in the following houses:
    /*Number of bedrooms either 3 or 4
    Bathrooms more than 3
    One Floor
    No waterfront
    Condition should be 3 at least
    Grade should be 5 at least
    Price less than 300000*/
    
select * from house_price_data
where bedrooms in (3,4)
and bathrooms > 3
and waterfront = 0
and condt >= 3
and grade >= 5
and price < 300000;

-- 12. Your manager wants to find out the list of properties whose prices are twice more than the average of all the properties in the database. 
-- Write a query to show them the list of such properties. You might need to use a sub query for this problem.

select * from house_price_data
where price > 2*(select avg(price) from house_price_data);

-- 13. Since this is something that the senior management is regularly interested in, create a view of the same query.
Create View twice_more_average as
select * from house_price_data
where price > 2*(select avg(price) from house_price_data);

select * from twice_more_average;

-- 14. Most customers are interested in properties with three or four bedrooms. What is the difference in average prices of the properties 
-- with three and four bedrooms?
select 
	bedrooms, 
	round(avg(price),2) as "average price", 
    round((select avg(price) from house_price_data
		where bedrooms = 4) - (select avg(price) from house_price_data
		where bedrooms = 3), 2) as diff 
from house_price_data
where bedrooms in (3,4)
group by bedrooms;

-- 15. What are the different locations where properties are available in your database? (distinct zip codes)
select distinct(zipcode) from house_price_data;

-- 16. Show the list of all the properties that were renovated.
select * from house_price_data
where yr_renovated > 0;

-- 17. Provide the details of the property that is the 11th most expensive property in your database.
select * from (select * from house_price_data order by price desc
limit 11) as sub1
where price = (select min(price) from (select * from house_price_data order by price desc
limit 11) as sub1);


-- 



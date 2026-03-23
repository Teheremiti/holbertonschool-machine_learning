-- Import database and display average temperature by city ordered by temp (desc)
SELECT city, AVG(value) AS avg_temp
FROM temperatures
GROUP BY city
ORDER by avg_temp DESC;
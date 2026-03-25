-- Shows by total rating
SELECT shows.title, SUM(show_rating.rate) AS rating
FROM tv_show_ratings AS show_rating
JOIN tv_shows AS shows
ON show_rating.show_id = shows.id
GROUP BY shows.title
ORDER BY rating DESC;
-- Shows by total rating
SELECT show.title, SUM(show_rating.rate) AS rating
FROM tv_show_ratings AS show_rating
JOIN tv_shows AS show
ON show_rating.show_id = show.id
GROUP BY show.title
ORDER BY rating DESC;
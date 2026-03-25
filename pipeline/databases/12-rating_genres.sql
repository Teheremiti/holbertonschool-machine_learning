-- Genres by total rating
SELECT genre.name, SUM(show_rating.rate) AS rating
FROM tv_genres AS genre
JOIN tv_show_genres AS show_genre
ON show_genre.genre_id = genre.id
JOIN tv_show_ratings AS show_rating
ON show_rating.show_id = show_genre.show_id
GROUP BY genre.name
ORDER BY rating DESC;
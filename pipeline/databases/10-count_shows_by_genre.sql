-- Shows per genre
SELECT genre.name AS genre, COUNT(show_genre.genre_id) AS number_of_shows
FROM tv_show_genres AS show_genre
JOIN tv_genres AS genre
ON genre.id = show_genre.genre_id
GROUP BY genre.name
ORDER BY number_of_shows DESC;
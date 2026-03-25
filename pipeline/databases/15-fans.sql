-- Fans by band origin
SELECT band.origin, SUM(band.fans) AS nb_fans
FROM metal_bands AS band
GROUP BY band.origin
ORDER BY nb_fans DESC;
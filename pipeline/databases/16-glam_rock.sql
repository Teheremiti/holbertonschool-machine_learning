-- Glam rock band lifespan
SELECT band.band_name, COALESCE(band.split, 2020) - band.formed AS lifespan
FROM metal_bands AS band
WHERE band.style LIKE '%Glam rock%'
ORDER BY lifespan DESC;
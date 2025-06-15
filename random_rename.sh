# Reemplazar "IMG_" por número aleatorio de 4 cifras seguido de "-"
for f in IMG_*; do
  rest="${f#IMG_}"

  # Generar número único
  while :; do
    num=$(( RANDOM % 9000 + 1000 ))
    new_name="${num}-${rest}"
    [[ ! -e "$new_name" ]] && break
  done

  echo "Renombrando: '$f' → '$new_name'"
  mv "$f" "$new_name"
done

cd plot_functions/tiny

for file in *.py
do
  mv "$file" "${file/pgd/auto}"
done
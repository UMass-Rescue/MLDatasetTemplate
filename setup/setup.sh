# Run commands from correct directory
cd ..

echo "Checking Image Storage Volume..."

# See if we need to create the image volume
if [[ $(docker volume inspect photoanalysisserver_images | grep "No such volume:") ]]; then
  echo "Creating Image Storage Volume..."
  docker volume create --name=photoanalysisserver_images
else
  echo "Image Storage Volume Already Exists!"
fi

# Build Docker Container
echo "Building Docker Container"
docker-compose build

# Start Docker Container
echo "Starting Docker Container..."
docker-compose up -d

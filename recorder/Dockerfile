FROM node:lts-buster

# Create app directory
WORKDIR /usr/src/app

# Copy dependency definitions
COPY package.json ./package.json
COPY package-lock.json ./package-lock.json
COPY postcss.config.js ./postcss.config.js
COPY tsconfig.json ./tsconfig.json
COPY webpack.common.js ./webpack.common.js
COPY webpack.dev.js ./webpack.dev.js
COPY webpack.prod.js ./webpack.prod.js

# Install dependencies
RUN npm set progress=false && npm config set depth 0 && npm i install
RUN npm ci

# Get all the code needed to run the app
COPY . .

# Expose the port the app runs in
EXPOSE 8080

# Serve the app
CMD ["npm", "run", "start"]
FROM nginx:alpine

# Copy website files
COPY website /usr/share/nginx/html

# Copy export directory if it exists
COPY export /usr/share/nginx/html/models

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
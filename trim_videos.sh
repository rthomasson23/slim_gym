#!/bin/bash

for file in ./static/videos/real_robot_vids/*.mp4; do
    duration=$(ffmpeg -i "$file" 2>&1 | grep "Duration" | cut -d ' ' -f 4 | sed s/,//)
    new_duration=$(echo $duration | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 - 0.1 }')
    ffmpeg -i "$file" -t $new_duration -c copy "${file%.mp4}_trimmed.mp4"
done


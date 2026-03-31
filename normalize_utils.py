


#This function get an aboslute path of a media file, check if exist
#Check if directory cache exist, otherwise it create it
#randomize a name usid cuid
# convert the media using ffmpeg into a 16kHz mono by averaging stereo
#write the normalized file into the disk
#return the absolute path (or None if error) to the caller.
def convert_and_store_normalized_audio_from_file(absolute_path:str):

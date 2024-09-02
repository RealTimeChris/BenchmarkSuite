/*
	MIT License

	Copyright (c) 2024 RealTimeChris

	Permission is hereby granted, free of charge, to any person obtaining a copy of this
	software and associated documentation files (the "Software"), to deal in the Software
	without restriction, including without limitation the rights to use, copy, modify, merge,
	publish, distribute, sublicense, and/or sell copies of the Software, and to permit
	persons to whom the Software is furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all copies or
	substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
	FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	DEALINGS IN THE SOFTWARE.
*/
/// https://github.com/RealTimeChris/jsonifier
/// Sep 17, 2024
#pragma once

#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include <glaze/glaze.hpp>

struct icon_emoji_data {
	std::optional<std::string> name;
	std::nullptr_t id;
};

struct permission_overwrite {
	std::string allow;
	int64_t type;
	std::string deny;
	std::string id;
};

struct channel_data {
	int64_t default_thread_rate_limit_per_user;
	int64_t default_auto_archive_duration;
	std::vector<permission_overwrite> permission_overwrites;
	int64_t rate_limit_per_user;
	int64_t video_quality_mode;
	int64_t total_message_sent;
	std::string last_pin_timestamp;
	std::optional<std::string> last_message_id;
	std::string application_id;
	int64_t message_count;
	int64_t member_count;
	std::vector<std::nullptr_t> applied_tags;
	std::string permissions;
	int64_t user_limit;
	icon_emoji_data icon_emoji;
	std::vector<std::nullptr_t> recipients;
	std::string parent_id;
	int64_t position;
	std::string guild_id;
	std::string owner_id;
	bool managed;
	int64_t bitrate;
	int64_t version;
	std::nullptr_t status;
	int64_t flags;
	std::nullptr_t topic;
	bool nsfw;
	int64_t type;
	std::string icon;
	std::string name;
	std::string id;
};

struct user_data {
	std::nullptr_t avatar_decoration_data;
	std::string discriminator;
	int64_t public_flags;
	int64_t premium_type;
	int64_t accent_color;
	std::optional<std::string> display_name;
	bool mfa_enabled;
	std::optional<std::string> global_name;
	std::string user_name;
	bool verified;
	bool system;
	std::nullptr_t locale;
	std::nullptr_t banner;
	std::optional<std::string> avatar;
	int64_t flags;
	std::string email;
	bool bot;
	std::string id;
};

struct member_data {
	std::nullptr_t communication_disabled_until;
	std::nullptr_t premium_since;
	std::string permissions;
	std::string joined_at;
	std::string guild_id;
	bool pending;
	std::nullptr_t avatar;
	int64_t flags;
	std::vector<std::string> roles;
	bool mute;
	bool deaf;
	user_data user;
	std::optional<std::string> nick;
};

struct tags_data {
	std::nullptr_t premium_subscriber;
	std::optional<std::string> bot_id;
};

struct role_data {
	std::nullptr_t unicode_emoji;
	bool mentionable;
	std::string permissions;
	int64_t position;
	bool managed;
	int64_t version;
	bool hoist;
	int64_t flags;
	int64_t color;
	tags_data tags;
	std::string name;
	std::nullptr_t icon;
	std::string id;
};

struct guild_data {
	std::nullptr_t latest_on_boarding_question_id;
	int64_t max_stage_video_channel_users;
	int64_t default_message_notifications;
	bool premium_progress_bar_enabled;
	int64_t approximate_presence_count;
	int64_t premium_subscription_count;
	std::string public_updates_channel_id;
	int64_t approximate_member_count;
	std::nullptr_t safety_alerts_channel_id;
	int64_t max_video_channel_users;
	int64_t explicit_content_filter;
	std::vector<std::nullptr_t> guild_scheduled_events;
	int64_t system_channel_flags;
	int64_t verification_level;
	std::nullptr_t inventory_settings;
	std::string widget_channel_id;
	std::string system_channel_id;
	std::string rules_channel_id;
	std::string preferred_locale;
	std::nullptr_t discovery_splash;
	std::nullptr_t vanity_url_code;
	bool widget_enabled;
	std::nullptr_t afk_channel_id;
	std::nullptr_t application_id;
	int64_t max_presences;
	int64_t premium_tier;
	int64_t member_count;
	std::vector<std::nullptr_t> voice_states;
	bool unavailable;
	int64_t afk_timeout;
	int64_t max_members;
	std::string permissions;
	std::nullptr_t description;
	int64_t nsfw_level;
	int64_t mfa_level;
	std::string joined_at;
	std::string discovery;
	std::string owner_id;
	std::nullptr_t hub_type;
	std::vector<std::string> stickers;
	std::vector<std::string> features;
	std::vector<channel_data> channels;
	std::vector<member_data> members;
	std::vector<std::nullptr_t> threads;
	std::string region;
	std::nullptr_t banner;
	std::nullptr_t splash;
	bool owner;
	bool large;
	int64_t flags;
	std::vector<role_data> roles;
	bool lazy;
	bool nsfw;
	std::string icon;
	std::string name;
	std::string id;
};

struct discord_message {
	int64_t op;
	int64_t s;
	guild_data d;
	std::string t;
};

template<> struct jsonifier::core<icon_emoji_data> {
	using value_type				 = icon_emoji_data;
	static constexpr auto parseValue = createValue<&value_type::name, &value_type::id>();
};

template<> struct jsonifier::core<permission_overwrite> {
	using value_type				 = permission_overwrite;
	static constexpr auto parseValue = createValue<&value_type::allow, &value_type::type, &value_type::deny, &value_type::id>();
};

template<> struct jsonifier::core<channel_data> {
	using value_type				 = channel_data;
	static constexpr auto parseValue = createValue<&value_type::default_thread_rate_limit_per_user, &value_type::default_auto_archive_duration, &value_type::permission_overwrites,
		&value_type::rate_limit_per_user, &value_type::video_quality_mode, &value_type::total_message_sent, &value_type::last_pin_timestamp, &value_type::last_message_id,
		&value_type::application_id, &value_type::message_count, &value_type::member_count, &value_type::applied_tags, &value_type::permissions, &value_type::user_limit,
		&value_type::icon_emoji, &value_type::recipients, &value_type::parent_id, &value_type::position, &value_type::guild_id, &value_type::owner_id, &value_type::managed,
		&value_type::bitrate, &value_type::version, &value_type::status, &value_type::flags, &value_type::topic, &value_type::nsfw, &value_type::type, &value_type::icon,
		&value_type::name, &value_type::id>();
};

template<> struct jsonifier::core<user_data> {
	using value_type				 = user_data;
	static constexpr auto parseValue = createValue<&value_type::avatar_decoration_data, &value_type::discriminator, &value_type::public_flags, &value_type::premium_type,
		&value_type::accent_color, &value_type::display_name, &value_type::mfa_enabled, &value_type::global_name, &value_type::user_name, &value_type::verified,
		&value_type::system, &value_type::locale, &value_type::banner, &value_type::avatar, &value_type::flags, &value_type::email, &value_type::bot, &value_type::id>();
};

template<> struct jsonifier::core<member_data> {
	using value_type = member_data;
	static constexpr auto parseValue =
		createValue<&value_type::communication_disabled_until, &value_type::premium_since, &value_type::permissions, &value_type::joined_at, &value_type::guild_id,
			&value_type::pending, &value_type::avatar, &value_type::flags, &value_type::roles, &value_type::mute, &value_type::deaf, &value_type::user, &value_type::nick>();
};

template<> struct jsonifier::core<tags_data> {
	using value_type				 = tags_data;
	static constexpr auto parseValue = createValue<&value_type::premium_subscriber, &value_type::bot_id>();
};

template<> struct jsonifier::core<role_data> {
	using value_type				 = role_data;
	static constexpr auto parseValue = createValue<&value_type::unicode_emoji, &value_type::mentionable, &value_type::permissions, &value_type::position, &value_type::managed,
		&value_type::version, &value_type::hoist, &value_type::flags, &value_type::color, &value_type::tags, &value_type::name, &value_type::icon, &value_type::id>();
};

template<> struct jsonifier::core<guild_data> {
	using value_type				 = guild_data;
	static constexpr auto parseValue = createValue<&value_type::latest_on_boarding_question_id, &value_type::max_stage_video_channel_users,
		&value_type::default_message_notifications, &value_type::premium_progress_bar_enabled, &value_type::approximate_presence_count, &value_type::premium_subscription_count,
		&value_type::public_updates_channel_id, &value_type::approximate_member_count, &value_type::safety_alerts_channel_id, &value_type::max_video_channel_users,
		&value_type::explicit_content_filter, &value_type::guild_scheduled_events, &value_type::system_channel_flags, &value_type::verification_level,
		&value_type::inventory_settings, &value_type::widget_channel_id, &value_type::system_channel_id, &value_type::rules_channel_id, &value_type::preferred_locale,
		&value_type::discovery_splash, &value_type::vanity_url_code, &value_type::widget_enabled, &value_type::afk_channel_id, &value_type::application_id,
		&value_type::max_presences, &value_type::premium_tier, &value_type::member_count, &value_type::voice_states, &value_type::unavailable, &value_type::afk_timeout,
		&value_type::max_members, &value_type::permissions, &value_type::description, &value_type::nsfw_level, &value_type::mfa_level, &value_type::joined_at,
		&value_type::discovery, &value_type::owner_id, &value_type::hub_type, &value_type::stickers, &value_type::features, &value_type::channels, &value_type::members,
		&value_type::threads, &value_type::region, &value_type::banner, &value_type::splash, &value_type::owner, &value_type::large, &value_type::flags, &value_type::roles,
		&value_type::lazy, &value_type::nsfw, &value_type::icon, &value_type::name, &value_type::id>();
};

template<> struct jsonifier::core<discord_message> {
	using value_type				 = discord_message;
	static constexpr auto parseValue = createValue<&value_type::op, &value_type::s, &value_type::d, &value_type::t>();
};

#if !defined(ASAN_ENABLED)

template<> struct glz::meta<icon_emoji_data> {
	using value_type			= icon_emoji_data;
	static constexpr auto value = object("name", &value_type::name, "id", &value_type::id);
};

template<> struct glz::meta<permission_overwrite> {
	using value_type			= permission_overwrite;
	static constexpr auto value = object("allow", &value_type::allow, "deny", &value_type::deny, "id", &value_type::id, "type", &value_type::type);
};

template<> struct glz::meta<channel_data> {
	using value_type			= channel_data;
	static constexpr auto value = object("permission_overwrites", &value_type::permission_overwrites, "last_message_id", &value_type::last_message_id,
		"default_thread_rate_limit_per_user", &value_type::default_thread_rate_limit_per_user, "applied_tags", &value_type::applied_tags, "recipients", &value_type::recipients,
		"default_auto_archive_duration", &value_type::default_auto_archive_duration, "status", &value_type::status, "last_pin_timestamp", &value_type::last_pin_timestamp, "topic",
		&value_type::topic, "rate_limit_per_user", &value_type::rate_limit_per_user, "icon_emoji", &value_type::icon_emoji, "total_message_sent", &value_type::total_message_sent,
		"video_quality_mode", &value_type::video_quality_mode, "application_id", &value_type::application_id, "permissions", &value_type::permissions, "message_count",
		&value_type::message_count, "parent_id", &value_type::parent_id, "member_count", &value_type::member_count, "owner_id", &value_type::owner_id, "guild_id",
		&value_type::guild_id, "user_limit", &value_type::user_limit, "position", &value_type::position, "name", &value_type::name, "icon", &value_type::icon, "version",
		&value_type::version, "bitrate", &value_type::bitrate, "id", &value_type::id, "flags", &value_type::flags, "type", &value_type::type, "managed", &value_type::managed,
		"nsfw", &value_type::nsfw);
};

template<> struct glz::meta<user_data> {
	using value_type			= user_data;
	static constexpr auto value = object("avatar_decoration_data", &value_type::avatar_decoration_data, "display_name", &value_type::display_name, "global_name",
		&value_type::global_name, "avatar", &value_type::avatar, "banner", &value_type::banner, "locale", &value_type::locale, "discriminator", &value_type::discriminator,
		"user_name", &value_type::user_name, "accent_color", &value_type::accent_color, "premium_type", &value_type::premium_type, "public_flags", &value_type::public_flags,
		"email", &value_type::email, "mfa_enabled", &value_type::mfa_enabled, "id", &value_type::id, "flags", &value_type::flags, "verified", &value_type::verified, "system",
		&value_type::system, "bot", &value_type::bot);
};

template<> struct glz::meta<member_data> {
	using value_type			= member_data;
	static constexpr auto value = object("communication_disabled_until", &value_type::communication_disabled_until, "premium_since", &value_type::premium_since, "nick",
		&value_type::nick, "avatar", &value_type::avatar, "roles", &value_type::roles, "permissions", &value_type::permissions, "joined_at", &value_type::joined_at, "guild_id",
		&value_type::guild_id, "user", &value_type::user, "flags", &value_type::flags, "pending", &value_type::pending, "deaf", &value_type::deaf, "mute", &value_type::mute);
};

template<> struct glz::meta<tags_data> {
	using value_type			= tags_data;
	static constexpr auto value = object("premium_subscriber", &value_type::premium_subscriber, "bot_id", &value_type::bot_id);
};

template<> struct glz::meta<role_data> {
	using value_type			= role_data;
	static constexpr auto value = object("unicode_emoji", &value_type::unicode_emoji, "icon", &value_type::icon, "permissions", &value_type::permissions, "position",
		&value_type::position, "name", &value_type::name, "mentionable", &value_type::mentionable, "version", &value_type::version, "id", &value_type::id, "tags",
		&value_type::tags, "color", &value_type::color, "flags", &value_type::flags, "managed", &value_type::managed, "hoist", &value_type::hoist);
};

template<> struct glz::meta<guild_data> {
	using value_type			= guild_data;
	static constexpr auto value = object("latest_on_boarding_question_id", &value_type::latest_on_boarding_question_id, "guild_scheduled_events",
		&value_type::guild_scheduled_events, "safety_alerts_channel_id", &value_type::safety_alerts_channel_id, "inventory_settings", &value_type::inventory_settings,
		"voice_states", &value_type::voice_states, "discovery_splash", &value_type::discovery_splash, "vanity_url_code", &value_type::vanity_url_code, "application_id",
		&value_type::application_id, "afk_channel_id", &value_type::afk_channel_id, "default_message_notifications", &value_type::default_message_notifications,
		"max_stage_video_channel_users", &value_type::max_stage_video_channel_users, "public_updates_channel_id", &value_type::public_updates_channel_id, "description",
		&value_type::description, "threads", &value_type::threads, "channels", &value_type::channels, "premium_subscription_count", &value_type::premium_subscription_count,
		"approximate_presence_count", &value_type::approximate_presence_count, "features", &value_type::features, "stickers", &value_type::stickers, "premium_progress_bar_enabled",
		&value_type::premium_progress_bar_enabled, "members", &value_type::members, "hub_type", &value_type::hub_type, "approximate_member_count",
		&value_type::approximate_member_count, "explicit_content_filter", &value_type::explicit_content_filter, "max_video_channel_users", &value_type::max_video_channel_users,
		"splash", &value_type::splash, "banner", &value_type::banner, "system_channel_id", &value_type::system_channel_id, "widget_channel_id", &value_type::widget_channel_id,
		"preferred_locale", &value_type::preferred_locale, "system_channel_flags", &value_type::system_channel_flags, "rules_channel_id", &value_type::rules_channel_id, "roles",
		&value_type::roles, "verification_level", &value_type::verification_level, "permissions", &value_type::permissions, "max_presences", &value_type::max_presences,
		"discovery", &value_type::discovery, "joined_at", &value_type::joined_at, "member_count", &value_type::member_count, "premium_tier", &value_type::premium_tier, "owner_id",
		&value_type::owner_id, "max_members", &value_type::max_members, "afk_timeout", &value_type::afk_timeout, "widget_enabled", &value_type::widget_enabled, "region",
		&value_type::region, "nsfw_level", &value_type::nsfw_level, "mfa_level", &value_type::mfa_level, "name", &value_type::name, "icon", &value_type::icon, "unavailable",
		&value_type::unavailable, "id", &value_type::id, "flags", &value_type::flags, "large", &value_type::large, "owner", &value_type::owner, "nsfw", &value_type::nsfw, "lazy",
		&value_type::lazy);
};

template<> struct glz::meta<discord_message> {
	using value_type			= discord_message;
	static constexpr auto value = object("t", &value_type::t, "d", &value_type::d, "op", &value_type::op, "s", &value_type::s);
};

#endif